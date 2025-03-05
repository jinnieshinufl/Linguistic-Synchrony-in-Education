from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import matplotlib.pyplot as plt

####### First code: document to document
def preprocess_text(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word.lower() for word in tokens if word.isalnum() or not word.isnumeric()]
    return ' '.join(filtered_tokens)

df = pd.read_csv('final_full_post_processed_equalized.csv')
distinct_filenames = df['filename'].unique()

component_levels = [50, 100, 300]
results_dfs = {level: [] for level in component_levels}

for filename in distinct_filenames:
    data = df[df['filename'] == filename]

    instructor_texts = data[data['speaker'] == 'instructor']['text_processed']
    instructor_df = pd.DataFrame({'instructor_texts': instructor_texts})

    student_texts = data[data['speaker'] == 'student']['text_processed']
    student_df = pd.DataFrame({'student_texts': student_texts})

    instructor_processed = [preprocess_text(doc) for doc in instructor_df['instructor_texts']]
    student_processed = [preprocess_text(doc) for doc in student_df['student_texts']]

    combined_instructor_text = ' '.join(instructor_processed)
    combined_student_text = ' '.join(student_processed)

    if not combined_instructor_text.strip() or not combined_student_text.strip():
            print(f"Skipping filename {filename} due to empty processed text.")
            continue
    
    all_processed = [combined_instructor_text, combined_student_text]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(all_processed)

    for level in component_levels:
        n_components = min(level, X.shape[1])
        svd = TruncatedSVD(n_components)
        X_svd = svd.fit_transform(X)

        # Separate the transformed vectors
        instructor_svd = X_svd[0].reshape(1, -1)
        student_svd = X_svd[1].reshape(1, -1)

        # Calculating similarity between the combined documents
        similarities = cosine_similarity(instructor_svd, student_svd)[0][0]
        results_dfs[level].append({'filename': filename, 'cosine_similarity': similarities})

X1_svd = X_svd[:len(instructor_processed)]
X2_svd = X_svd[len(instructor_processed):]

#Calculating pairwise
similarities = []
for i in range(min(X1_svd.shape[0], X2_svd.shape[0])):
    sim = cosine_similarity(X1_svd[i].reshape(1, -1), X2_svd[i].reshape(1, -1))[0][0]
    similarities.append({'Instructor_Index': i, 'Student_Index': i, 'Cosine_Similarity': sim})    

similarity_df = pd.DataFrame(similarities)
print(similarity_df)
similarity_df.to_csv('pairwise_LSS.csv', index=True)

writer = pd.ExcelWriter('doc_doc_LSS.xlsx', engine='xlsxwriter')
for level, results in results_dfs.items():
    df = pd.DataFrame(results)
    sheet_name = f'{level} components'
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    
writer.close()
print('Similarities calculation complete and saved to Excel.')

# Plotting the line graph
plt.figure(figsize=(10, 6))
plt.plot(similarity_df['Instructor_Index'], similarity_df['Cosine_Similarity'], marker='o', linestyle='-')
plt.xlabel('Index')
plt.ylabel('Cosine Similarity')
plt.title('Cosine Similarity between Instructor and Student Utterances')
plt.grid(True)
plt.show()

####### Second code: term to term
def preprocess_text(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word.lower() for word in tokens if word.isalnum() or not word.isnumeric()]
    return ' '.join(filtered_tokens)

# Load the dataset
df = pd.read_csv('final_full_post_processed_equalized.csv')
distinct_filenames = df['filename'].unique()

# Component levels to evaluate
component_levels = [50, 100, 300]

# Initialize a dictionary for dataframes at each component level
results_dfs = {level: pd.DataFrame() for level in component_levels}

# Process each filename
for filename in distinct_filenames:
    data = df[df['filename'] == filename]

    instructor_texts = data[data['speaker'] == 'instructor']['text_processed'].tolist()
    student_texts = data[data['speaker'] == 'student']['text_processed'].tolist()

    instructor_processed = [preprocess_text(doc) for doc in instructor_texts]
    student_processed = [preprocess_text(doc) for doc in student_texts]

    all_texts = instructor_processed + student_processed

    # Vectorization
    vectorizer = TfidfVectorizer()
    all_X = vectorizer.fit_transform(all_texts)

    for level in component_levels:
        n_components = min(level, all_X.shape[1]) 
        svd = TruncatedSVD(n_components=n_components)
        all_X_svd = svd.fit_transform(all_X)

        # Separate the transformed vectors
        X1_svd = all_X_svd[:len(instructor_processed)]
        X2_svd = all_X_svd[len(instructor_processed):]

        # Calculating pairwise similarities
        similarities = []
        for i in range(min(len(instructor_processed), len(student_processed))):
            sim = cosine_similarity(X1_svd[i].reshape(1, -1), X2_svd[i].reshape(1, -1))[0][0]
            similarities.append({'Filename': filename, 'Instructor_Index': i, 'Student_Index': i, 'LSS index': sim, 'Components': n_components})

        # Append to the dataframe for this component level
        if not results_dfs[level].empty:
            results_dfs[level] = pd.concat([results_dfs[level], pd.DataFrame(similarities)])
        else:
            results_dfs[level] = pd.DataFrame(similarities)

# Create a Pandas Excel writer 
writer = pd.ExcelWriter('term_term_LSS.xlsx', engine='xlsxwriter')

# Write each DataFrame to a different sheet based on component level
for level, df in results_dfs.items():
    sheet_name = f'{level}_components'
    df.to_excel(writer, sheet_name=sheet_name, index=False)

writer.close()
print('Pairwise similarities calculation complete and saved to Excel.')