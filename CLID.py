import ot
import pandas as pd
from nltk.corpus import stopwords
import gensim.downloader as api
import matplotlib.pyplot as plt
import numpy as np
# from nltk import download
# download('stopwords') 


stop_words = stopwords.words('english')

def preprocess(sentence):
    # return [w for w in sentence.lower().split() if w not in stop_words]
    return sentence


df = pd.read_csv('processed_data_tscc.csv')
distinct_filenames = df['chat_id'].unique()

model = api.load('word2vec-google-news-300')

wmd_results = []
summary_results = []

for filename in distinct_filenames:
    data = df[df['chat_id'] == filename]
    print(f"Currently Processing '{filename}'")

    instructor_texts = data[data['role'] == 'teacher']['anonymised']
    instructor_texts = instructor_texts.reset_index(drop=True)
    instructor_df = pd.DataFrame({'instructor_texts': instructor_texts})

    student_texts = data[data['role'] == 'student']['anonymised']
    student_texts = student_texts.reset_index(drop=True)
    student_df = pd.DataFrame({'student_texts': student_texts})

    k = 2 #context length
    distances = []

    min_length = min(len(instructor_df), len(student_df))

    for i in range(len(instructor_texts)):
        sentence_instructor = ' '.join(preprocess(instructor_texts[i]))
        
        # Consider the next k utterances from the student
        min_distance = float('inf')
        for j in range(i, min(i + k, len(student_texts))):
            sentence_student = ' '.join(preprocess(student_texts[j]))
            distance = model.wmdistance(sentence_instructor, sentence_student)
            if distance < min_distance:
                min_distance = distance
        
        distances.append(min_distance)

        print(f'Index {i}: minimum distance = {min_distance:.4f}')

    #Calculating session level linguistic coordination
    valid_distances = [d for d in distances if np.isfinite(d)]
    uclid = sum(valid_distances) / len(valid_distances)
    print(f'Session-level measure (uCLiD): {uclid:.4f}')

    for index, distance in enumerate(distances):
        wmd_results.append([filename, index, distance])

    distances_df = pd.DataFrame({'index': range(len(instructor_texts)), 'wmd_distance': distances})
    # distances_df.to_csv(f'{specific_filenames}_wmd_distances.csv', index=False)

    # Calculate normalization factor
    N = len(instructor_texts)
    M = len(student_texts)
    pairwise_wmd_instructor = []
    pairwise_wmd_student = []
    pairwise_wmd_cross = []

    # Within instructor's utterances (A_i, A_j)
    if N > 1:
        for i in range(N):
            for j in range(i + 1, N):
                sentence_instructor_i = ' '.join(preprocess(instructor_texts[i]))
                sentence_instructor_j = ' '.join(preprocess(instructor_texts[j]))
                distance = model.wmdistance(sentence_instructor_i, sentence_instructor_j)
                if np.isfinite(distance):
                    pairwise_wmd_instructor.append(distance)

    # Within student's utterances (B_i, B_j)
    if M > 1:
        for i in range(len(student_texts)):
            for j in range(i + 1, len(student_texts)):
                sentence_student_i = ' '.join(preprocess(student_texts[i]))
                sentence_student_j = ' '.join(preprocess(student_texts[j]))
                distance = model.wmdistance(sentence_student_i, sentence_student_j)
                if np.isfinite(distance):
                    pairwise_wmd_instructor.append(distance)

    # Between instructor's and student's utterances (A_i, B_j)
    for i in range(len(instructor_texts)):
        for j in range(len(student_texts)):
            sentence_instructor = ' '.join(preprocess(instructor_texts[i]))
            sentence_student = ' '.join(preprocess(student_texts[j]))
            distance = model.wmdistance(sentence_instructor, sentence_student)
            if np.isfinite(distance):
                pairwise_wmd_instructor.append(distance)

    # Calculate alpha
    alpha_instructor = (2 / (N * (N - 1))) * sum(pairwise_wmd_instructor) if N > 1 else 0
    alpha_student = (2 / (M * (M - 1))) * sum(pairwise_wmd_student) if M > 1 else 0
    alpha_cross = (2 / (N * M)) * sum(pairwise_wmd_cross) if N > 0 and M > 0 else 0
    alpha = alpha_instructor + alpha_student + alpha_cross

    if alpha == 0:
        continue 

    print(f'Normalization factor (alpha): {alpha:.4f}')

    nclid = uclid/alpha
    print(f'nCLiD: {nclid:.4f}')
    summary_results.append([filename, uclid, nclid])


with pd.ExcelWriter('og_tscc_wmd_results.xlsx') as writer:
    wmd_df = pd.DataFrame(wmd_results, columns=['filename', 'index', 'wmd_distance'])
    wmd_df.to_excel(writer, sheet_name='Local coordination', index=False)

    summary_df = pd.DataFrame(summary_results, columns=['filename', 'uclid', 'nclid'])
    summary_df.to_excel(writer, sheet_name='Session level', index=False)


# plt.plot(distances_df['index'], distances_df['wmd_distance'], marker='o')
# plt.xlabel('Index')
# plt.ylabel('WMD Distance')
# plt.title('WMD Distances Over Instructor Utterances')
# plt.show()

