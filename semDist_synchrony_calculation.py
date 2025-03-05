from scipy.spatial.distance import cosine
import pandas as pd
import numpy as np

# original file of the dataset
broward_file_path = #data file path
broward = pd.read_csv(broward_file_path)

tscc_file_path = #data file path
tscc = pd.read_csv(tscc_file_path)

##### Glove and Fasttext
# Load the embeddings datasets
broward_gft = #load embedding file
tscc_gft = #load embedding file

# Add glove and fasttext embeddings to the broward DataFrame
broward['glove'] = broward_gft['glove']
broward['fasttext'] = broward_gft['fasttext']

# Add glove and fasttext embeddings to the tscc DataFrame
tscc['glove'] = tscc_gft['glove']
tscc['fasttext'] = tscc_gft['fasttext']

def convert_string_to_list(s):
    return list(map(float, s.strip('[]').split()))

# Apply the function to the 'glove' column
broward['glove'] = broward['glove'].apply(convert_string_to_list)
broward['fasttext'] = broward['fasttext'].apply(convert_string_to_list)

tscc['glove'] = tscc['glove'].apply(convert_string_to_list)
tscc['fasttext'] = tscc['fasttext'].apply(convert_string_to_list)

##### Calculating cosine distance #####
from scipy.spatial.distance import cosine

def calculate_semDist(instructor_utterance, student_utterance):
    return cosine(instructor_utterance, student_utterance)

# Initialize lists to store the results
results_broward = []
results_tscc = []

# Loop over each filename in Broward dataset and calculate cosine distance
for filename in broward['filename'].unique():
    data = broward[broward['filename'] == filename]
    
    for i in range(0, len(data)-1, 1):  
        utt_1_glove = data.iloc[i]['glove']
        utt_2_glove = data.iloc[i+1]['glove']
        utt_1_fasttext = data.iloc[i]['fasttext']
        utt_2_fasttext = data.iloc[i+1]['fasttext']
    
        # Calculate cosine distances
        glove_distance = calculate_semDist(utt_1_glove, utt_2_glove)
        fasttext_distance = calculate_semDist(utt_1_fasttext, utt_2_fasttext)
        
        # Append results to the list for Broward
        results_broward.append({
            'filename': filename,
            'glove_cosine_distance': glove_distance,
            'fasttext_cosine_distance': fasttext_distance
        })

# Loop over each chat_id in TSCC dataset and calculate cosine distance
for chat_id in tscc['chat_id'].unique():
    # Filter by chat_id
    data = tscc[tscc['chat_id'] == chat_id]
    
    for i in range(0, len(data)-1, 1): 
        utt_1_glove = data.iloc[i]['glove']
        utt_2_glove = data.iloc[i+1]['glove']
        utt_1_fasttext = data.iloc[i]['fasttext']
        utt_2_fasttext = data.iloc[i+1]['fasttext']
    
        # Calculate cosine distances
        glove_distance = calculate_semDist(utt_1_glove, utt_2_glove)
        fasttext_distance = calculate_semDist(utt_1_fasttext, utt_2_fasttext)
        
        # Append results to the list for TSCC
        results_tscc.append({
            'chat_id': chat_id,
            'glove_cosine_distance': glove_distance,
            'fasttext_cosine_distance': fasttext_distance
        })

# Convert the results lists to DataFrames
df_broward_results = pd.DataFrame(results_broward)
df_tscc_results = pd.DataFrame(results_tscc)

# Save the results to CSV files
df_broward_results.to_csv('broward_semDist_gft.csv', index=False)
df_tscc_results.to_csv('tscc_semDist_gft.csv', index=False)

# Print out the first few rows of the results to verify
print("Broward Cosine Distance Results:")
print(df_broward_results.head())

print("TSCC Cosine Distance Results:")
print(df_tscc_results.head())

import os
##### BERT semDist
# Load BERT embeddings datasets
def calculate_semDist(utt1, utt2):
    return cosine(utt1, utt2)

dir_path = #directory to folder containing npy embedding files
tscc_elmo_sem_dist = pd.DataFrame(columns=['chat_id', 'semDist'])
files = [f for f in os.listdir(dir_path) if f.endswith('.npy')]

# Process each file
for file in files:
    # Load embeddings from file
    embeddings = np.load(os.path.join(dir_path, file))
    filename = file.split('_')[0]  # Extract chat_id/filename from filename 

    new_rows = []
    # Calculate semantic distances between consecutive embeddings
    for i in range(len(embeddings) - 1):
        print(f"Processing index {i+1}/{len(embeddings)-1} for chat_id {filename}")
        utterance_1 = embeddings[i]
        utterance_2 = embeddings[i+1]
        semDist = calculate_semDist(utterance_1, utterance_2)
        new_rows.append({'chat_id': filename, 'semDist': semDist})
        
    tscc_elmo_sem_dist = pd.concat([tscc_elmo_sem_dist, pd.DataFrame(new_rows)], ignore_index=True)


# Save to CSV
tscc_elmo_sem_dist.to_csv(dir_path, index=False)
print("All distances calculated and saved.")

tscc1_bert = #load npy embedding file
print("Shape of tscc1_bert:", tscc1_bert.shape)
tscc2_bert = #load npy embedding file
print("Shape of tscc2_bert:", tscc2_bert.shape)

# broward_bert = np.concatenate((broward1_bert, broward2_bert, broward3_bert), axis=0)
tscc_bert = np.concatenate((tscc1_bert, tscc2_bert), axis=0)

def calculate_semDist(utt1, utt2):
    return cosine(utt1, utt2)

# broward_bert_sem_dist = pd.DataFrame(columns=['filename', 'semDist'])

# for filename in broward['filename'].unique():
#     indices = broward[broward['filename'] == filename].index
#     for i in range(len(indices) - 1):
#         idx = indices[i]
#         utterance_1 = np.mean(broward_bert[idx], axis=0)
#         utterance_2 = np.mean(broward_bert[idx+1], axis=0)
#         semDist = calculate_semDist(utterance_1, utterance_2)
#         broward_bert_sem_dist = broward_bert_sem_dist.append({'filename': filename, 'semDist': semDist}, ignore_index=True)

tscc_bert_sem_dist = pd.DataFrame(columns=['filename', 'semDist'])
for chat_id in tscc['chat_id'].unique:
    indices = tscc[tscc['chat_id'] == chat_id].index
    for i in range(len(indices) - 1):
        print(f"Processing index {i+1}/{len(indices)-1} for chat_id {chat_id}")
        idx = indices[i]
        utterance_1 = np.mean(tscc_bert[idx], axis=0)
        utterance_2 = np.mean(tscc_bert[idx+1], axis=0)
        semDist = calculate_semDist(utterance_1, utterance_2)
        tscc_bert_sem_dist.append(semDist)

# broward_bert_semDist = pd.DataFrame(broward_bert_sem_dist, columns=['semDist'])
tscc_bert_semDist = pd.DataFrame(tscc_bert_sem_dist, columns=['semDist'])
# broward_bert_semDist.to_csv('broward_bert_semDist.csv', index=False)
tscc_bert_semDist.to_csv('tscc_bert_semDist.csv', index=False)

###### elmo semDist
# load elmo embeddings
broward_elmo = # load embedding file
tscc_elmo = # load embedding file

broward_elmo_sem_dist = []
for filename in broward['filename'].unique():
    indices = broward[broward['filename'] == filename].index
    for i in range(len(indices) - 1):
        idx = indices[i]
        utterance_1 = broward_elmo[idx]
        utterance_2 = broward_elmo[idx+1]
        semDist = calculate_semDist(utterance_1, utterance_2)
        broward_elmo_sem_dist.append(semDist)

tscc_elmo_sem_dist = []
for chat_id in tscc['chat_id'].unique():
    indices = tscc[tscc['chat_id'] == chat_id].index
    for i in range(len(indices) - 1):
        idx = indices[i]
    utterance_1 = tscc_elmo[idx]
    utterance_2 = tscc_elmo[idx+1]
    semDist = calculate_semDist(utterance_1, utterance_2)
    tscc_elmo_sem_dist.append(semDist)

broward_elmo_semDist = pd.DataFrame(broward_elmo_sem_dist, columns=['semDist'])
tscc_elmo_semDist = pd.DataFrame(tscc_elmo_sem_dist, columns=['semDist'])

broward_elmo_semDist.to_csv('broward_elmo_semDist.csv', index=False)
tscc_elmo_semDist.to_csv('tscc_elmo_semDist.csv', index=False)
