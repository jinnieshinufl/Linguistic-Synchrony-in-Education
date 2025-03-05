import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from gensim.models import FastText
import numpy as np
import random
import torch
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import os 

# Reading Datasets
data = pd.read_excel("cleaned_data_processed.xlsx", sheet_name="normalized")



##### Method 2: Contextualized Sentence Embeddings #####
#### BERT 
save_dir = r'C:\Users\pauli\OneDrive - University of Florida\Documents\Linguistic Synchrony\USF Dataset\Cleaned Data\new embeddings\data_xlmr_embeddings_new'
os.makedirs(save_dir, exist_ok=True)

random_seed = 42
random.seed(random_seed)

torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

# broward_tokens_10 = broward_tokens[:10]
# tscc_tokens_10 = tscc_tokens[:10]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")


# Iterate over each filename group
for filename, group in data.groupby('filename'):
    print(f'Current Filename: {filename}')
    data_tokens = group['talk'].astype(str)

    # encoding data
    data_encoding = tokenizer.batch_encode_plus(
        data_tokens,                    
        padding=True,              
        truncation=True,           
        return_tensors='pt',      
        add_special_tokens=True   
    )

    batch_size = 10

    # Prepare input tensors
    data_input_ids = data_encoding['input_ids']
    data_attention_mask = data_encoding['attention_mask']

    # Pre-define lists to store embeddings
    data_bert_embeddings = []

    # Process dataset in batches
    with torch.no_grad():
        for i in range(0, len(data_input_ids), batch_size):
            data_input_ids_batch = data_input_ids[i:i + batch_size]
            data_attention_mask_batch = data_attention_mask[i:i + batch_size]
            
            # embeddings
            data_outputs = model(data_input_ids_batch, attention_mask=data_attention_mask_batch)
            data_bert_embeddings_temp = data_outputs.last_hidden_state

            data_bert_embeddings.append(data_bert_embeddings_temp.numpy())
            print('Finished processing batch:', i // batch_size, 'Range of indices:', i, 'to', i + batch_size - 1)

    # flatten the list of batches into a single list of embeddings
    data_bert_embeddings = np.concatenate(data_bert_embeddings, axis=0)

        # Ensure the lengths match
        # assert len(data_bert_embeddings) == 10

    # Construct full path for the file to save
    file_path = os.path.join(save_dir, f'{filename}_xlmr_embeddings.npy')

    # embeddings to numpy arrays and save
    np.save(file_path, data_bert_embeddings)
    print(f'Saved embeddings for filename: {filename}')


