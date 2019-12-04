import os
import re
import json

import numpy as np

from sklearn import preprocessing
#%%
def normalize_mention(mention):
    mention = re.sub(r'[,.;@#?!&$-:/]+\ *', ' ', mention)
    return mention.strip()
#%%
# Read a1 files
with open('configs.json') as f:
    configs = json.load(f)

train_path = configs['train']
word_embeddings_file_path = configs['word_embeddings_100']
mention_embeddings_file_path = configs['mention_embeddings_100']
word_vector_dim = 100
#%%
a1_files = [f for f in os.listdir(train_path) if f.endswith('.a1')]
print('Extracting mentions')
# Filter a1 files by Habitats
habitat_mentions = []           
for a1_file in a1_files:
    with open(train_path + a1_file, encoding='latin5') as a1:
        annotations = [l.split('\t') for l in a1.readlines()]
        habitat_mentions = habitat_mentions + \
                        [annotation[-1] for annotation in annotations if 'Habitat' in annotation[1]]

# normalize mentions and get unique mentions only
habitat_mentions = list(set([normalize_mention(mention) for mention in habitat_mentions]))
#%%
print('Loading word vectors')
with open(word_embeddings_file_path) as embedding_file:
    word_vectors = json.load(embedding_file)
#%%
print('Creating mention embeddings')
mention_to_embedding = {}
cant_find = []
lower_counts = 0
for mention in habitat_mentions:
    mention_embedding = np.zeros((word_vector_dim))
    word_count_in_mention = 0
    for word in mention.split():
        word = word.strip()
        if word in word_vectors:
            mention_embedding = mention_embedding + word_vectors[word]
            word_count_in_mention = word_count_in_mention + 1
        elif word.lower() in word_vectors:
            mention_embedding = mention_embedding + word_vectors[word.lower()]
            word_count_in_mention = word_count_in_mention + 1
            lower_counts = lower_counts + 1
        else:
            cant_find.append(word)
    
    if word_count_in_mention > 0:
        mention_embedding = mention_embedding / word_count_in_mention            
        mention_to_embedding[mention] = preprocessing.normalize([mention_embedding], norm='l2')[0].tolist()
#%%
print('Dumping mention embeddings')
with open(mention_embeddings_file_path, 'w') as f:
    json.dump(mention_to_embedding, f)
        
        