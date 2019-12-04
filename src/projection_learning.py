import os
import re
import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#%%
def normalize_mention(mention):
    mention = re.sub(r'[,.;@#?!&$-:/]+\ *', ' ', mention)
    return mention.strip()
#%%
with open('configs.json') as f:
    configs = json.load(f)

train_path = configs['train']
mention_embeddings_path = configs['mention_embeddings_100']
node_embeddings_path = configs['node_embeddings']
#%%
a1_files = [f for f in os.listdir(train_path) if f.endswith('.a1')]
#mention_to_node_id = {}
mentions_and_node_ids = []          
for a1_file in a1_files:
    with open(train_path + a1_file, encoding='latin5') as a1:
        annotations = [l.split('\t') for l in a1.readlines()]
        tag_to_mention = {annotation[0]: normalize_mention(annotation[-1]) for annotation in annotations if 'Habitat' in annotation[1]}
    
    a2_file = a1_file.replace('.a1', '.a2')
    with open(train_path + a2_file) as a2:
        annotations = [l.split() for l in a2.readlines()]
        
    tag_to_node_id = {annotation[2].split(':')[-1]: 'OBT:' + annotation[3].split(':')[-1] for annotation in annotations 
                if annotation[1] == 'OntoBiotope'}
    
    # there are some conflicting mappings. Children are mapped to 2146 and 2216.
    # for know we do not fix them.
#    for tag, mention in tag_to_mention.items():
#        node_id = tag_to_node_id[tag]
#        if mention in mention_to_node_id:
#            if node_id != mention_to_node_id[mention]:
#                print(mention, node_id, 'UPS!')
#        else:
#            mention_to_node_id[mention] = node_id
    for tag, mention in tag_to_mention.items():
        node_id = tag_to_node_id[tag]
        mentions_and_node_ids.append((mention, node_id))
#%%
# get unique mappings
mentions_and_node_ids = list(set(mentions_and_node_ids))
#%%
with open(mention_embeddings_path) as f:
    mention_embeddings = json.load(f)
#%%
node_embeddings = {}
with open(node_embeddings_path) as f:
    lines = f.readlines()[1:]
    for line in lines:
        tokens = line.split()
        node_embeddings[tokens[0]] = [float(token) for token in tokens[1:]]
#%%
X_train, Y_train = [], []
for mention, node_id in mentions_and_node_ids:
    X_train.append(mention_embeddings[mention])
    Y_train.append(node_embeddings[node_id])

X_train = np.array(X_train)
Y_train = normalize(Y_train)
#%%
model = Sequential()
model.add(Dense(100, activation=None))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])
history = model.fit(X_train, Y_train, epochs=300, verbose=0).history

plt.plot(history['loss'])
plt.show()
#%%







