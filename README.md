# Linguistic-Synchrony-in-Education
This repository includes the synchrony measures that were compared in Shin &amp; Aguinalde (under review)

## Overview
This repository contains the code and resources associated with the study titled _The Computational Measurement of Linguistic Synchrony in Online Education Environments: A Comparative Analysis of Various Computational Methods_. This study systematically compares several computational approaches for measuring linguistic synchrony in one-to-one educational tutoring environments, focusing on Algebra and Language tutoring conversations.

## Methods
The following computational methods are implemented in this repository:

- ALIGN (Duran et al., 2019): Measures lexical, syntactic, and semantic alignment between conversational partners using cosine similarity of word frequency vectors and part-of-speech (POS) distributions.
- Conversational Linguistic Distance (CLiD): Computes Word Mover's Distance (WMD) to assess semantic alignment between conversational partners.
- Jensen-Shannon Divergence over Parts of Speech (JSDuPOS): Measures syntactic synchrony using Jensen-Shannon Divergence across POS tag distributions.
- Cross-Recurrence Quantification Analysis (CRQA): Analyzes temporal synchrony by constructing recurrence plots to capture alignment patterns over time.
- Linguistic Inquiry and Word Count (LIWC): Analyzes psychological and linguistic dimensions of texts, including emotional tone and cognitive processes, to evaluate pragmatic synchrony.
- Latent Semantic Analysis (LSA): Computes semantic similarity between texts using high-dimensional semantic space representations.
- Transformer-based Models (BERT, GloVe, FastText): Utilizes word embeddings to capture the semantic meaning and context of utterances, calculating synchrony through cosine similarity.

## Installation
Clone the repository:
```{python}
git clone https://github.com/your-repo/synchrony-computation.git
cd synchrony-computation
```
Set up the virtual environment and install required packages:
```{python}
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
``` 
## Data
We requested for data from the Teacher-Student Chat Room Corpus (TSCC; Caines et al., 2020; 2022).
https://paperswithcode.com/paper/the-teacher-student-chatroom-corpus 

## Usage
```{python}
# Preprocessing
python data_preprocessing.py --input data/raw --output data/processed

# Synchrony measures
python compute_synchrony.py --method ALIGN --input data/processed --output results/align
``` 

