import pandas as pd
import numpy as np
from collections import Counter
from scipy.spatial.distance import jensenshannon
import stanza
import matplotlib.pyplot as plt

nlp = stanza.Pipeline(lang='en', processors='tokenize,pos', use_gpu=True)
def POS_tagged(texts):
    tagged_texts = []
    for text in texts:
        doc = nlp(texts)
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

def calculate_js_divergence(dist1, dist2):
    if not dist1 or not dist2:
        return 1.0
    all_tags = set(dist1.keys()).union(set(dist2.keys()))
    p = np.array([dist1.get(tag, 0) for tag in all_tags])
    q = np.array([dist2.get(tag, 0) for tag in all_tags])
    js_div = jensenshannon(p, q, base=2)
    return js_div

def convert_float_to_str(item):
    if isinstance(item, float):
        return str(item)
    return item

#Computing synchrony
def compute_synchrony(instructor, student):
    stop_val = min(len(instructor), len(student)) - 1
    js_vals = []
    instructor_dist = []
    student_dist = []
    index = 0

    while index <= stop_val:
        instructor_text = instructor.loc[index]
        instructor_text = POS_tagged(instructor_text)
        instructor_text = compute_pos_distribution(instructor_text)
        instructor_dist.append(instructor_text)

        student_text = student.loc[index]
        student_text = POS_tagged(student_text)
        student_text = compute_pos_distribution(student_text)
        student_dist.append(student_text)
        ling_sim = calculate_js_divergence(instructor_text,student_text)
        js_vals.append(ling_sim)

        index += 1

        if index==stop_val:
            break

        # Convert lists of dictionaries to DataFrames
        instructor_df = pd.DataFrame(instructor_dist)
        student_df = pd.DataFrame(student_dist)

        # Fill NaN values that appear if some POS tags are missing in some distributions
        instructor_df.fillna(0, inplace=True)
        student_df.fillna(0, inplace=True)

        # Save to Excel
        with pd.ExcelWriter('POS_Distributions.xlsx') as writer:
            instructor_df.to_excel(writer, sheet_name='Instructor POS Distributions', index=False)
            student_df.to_excel(writer, sheet_name='Student POS Distributions', index=False)
    return js_vals, instructor_dist, student_dist


# Load the dataset
df = pd.read_csv('full_post_processed_equalized.csv')
df = df[df['filename'] == 'f3b69512-0b6a-49fa-bac7-df278420cedb']

# Separate texts by speaker
instructor_texts = df[df['speaker'] == 'instructor']['text_processed']
instructor_texts = instructor_texts.reset_index(drop=True).apply(convert_float_to_str)
student_texts = df[df['speaker'] == 'student']['text_processed']
student_texts = student_texts.reset_index(drop=True).apply(convert_float_to_str)

print(len(instructor_texts))
print(len(student_texts))

synch, instructor_distributions, student_distributions = compute_synchrony(instructor_texts, student_texts)
synchrony = sum(synch) / len(synch)
print("Linguistic Similarity Values: ", synch)
print("Linguistic Synchrony: ", synchrony)

# Plotting the values
indexes = list(range(len(synch)))
plt.figure(figsize=(10, 5)) 
plt.plot(indexes, synch, marker='o') 

# Adding labels and title
plt.title('Linguistic Similarity over the Session')
plt.xlabel('Index')
plt.ylabel('Similarity (JS divergence)')

# Show the plot
plt.grid(True) 
plt.show()
