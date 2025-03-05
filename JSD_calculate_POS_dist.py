import pandas as pd
import numpy as np
from collections import Counter
from scipy.spatial.distance import jensenshannon
import stanza
import matplotlib.pyplot as plt
import os

nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos', use_gpu=True)

def POS_tagged(texts):
    tagged_texts = []
    for text in texts:
        doc = nlp(text)
        tagged_sentences = [(word.text, word.upos) for sent in doc.sentences for word in sent.words]
        tagged_texts.append(tagged_sentences)
    return tagged_texts

def compute_pos_distribution(tagged_texts):
    if not tagged_texts:
        return {}
    pos_counts = Counter(tag for sentence in tagged_texts for _, tag in sentence)
    total_counts = sum(pos_counts.values())
    pos_distribution = {tag: count / total_counts for tag, count in pos_counts.items()}
    return pos_distribution

def convert_float_to_str(item):
    if isinstance(item, float):
        return str(item)
    return item

# Load the dataset
df = pd.read_csv('tscc_full_post_processed_equalized.csv')
distinct_filenames = df['chat_id'].unique()

save_folder = 'POS_distributions_update'
os.makedirs(save_folder, exist_ok=True)

for file in distinct_filenames:
    data = df[df['chat_id'] == file]

    # Separate texts by speaker
    instructor_texts = data[data['role'] == 'teacher']['text_processed']
    instructor_texts = instructor_texts.reset_index(drop=True).apply(convert_float_to_str)
    student_texts = data[data['role'] == 'student']['text_processed']
    student_texts = student_texts.reset_index(drop=True).apply(convert_float_to_str)

    print(f"Processing filename: {file}")
    print("Num of instructor text: ", len(instructor_texts))
    print("Num of student text: ", len(student_texts))

    # Compute POS distributions for both instructor and student
    instructor_tagged = POS_tagged(instructor_texts)
    student_tagged = POS_tagged(student_texts)

    instructor_distribution = [compute_pos_distribution([sentence]) for sentence in instructor_tagged]
    student_distribution = [compute_pos_distribution([sentence]) for sentence in student_tagged]

    instructor_df = pd.DataFrame(instructor_distribution).fillna(0)
    student_df = pd.DataFrame(student_distribution).fillna(0)

    # Save the instructor POS distribution
    instructor_output_file = os.path.join(save_folder, f"{file}_instructor_pos_distribution.csv")
    instructor_df.to_csv(instructor_output_file, index=False)

    # Save the student POS distribution
    student_output_file = os.path.join(save_folder, f"{file}_student_pos_distribution.csv")
    student_df.to_csv(student_output_file, index=False)

    print(f"POS distributions saved for {file} (instructor and student separately)")