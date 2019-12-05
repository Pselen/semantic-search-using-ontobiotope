import os
import re
import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx

from sklearn.preprocessing import normalize
from sklearn.metrics import precision_score

from tensorflow.keras.models import load_model
#%%
def normalize_mention(mention):
    mention = re.sub(r'[,.;@#?!&$-:/]+\ *', ' ', mention)
    return mention.strip()
#%%
with open('configs.json') as f:
    configs = json.load(f)
#%%
model = load_model(configs['model_path'])
test_path = configs['dev']
mention_embeddings_path = configs['mention_embeddings_100']
node_embeddings_path = configs['node_embeddings']
#%%
a1_files = [f for f in os.listdir(test_path) if f.endswith('.a1')]
#mention_to_node_id = {}
mentions_and_node_ids = []          
for a1_file in a1_files:
    with open(test_path + a1_file, encoding='latin5') as a1:
        annotations = [l.split('\t') for l in a1.readlines()]
        tag_to_mention = {annotation[0]: normalize_mention(annotation[-1]) for annotation in annotations if 'Habitat' in annotation[1]}
    
    a2_file = a1_file.replace('.a1', '.a2')
    with open(test_path + a2_file) as a2:
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
print('Extracted mention node id pairs')
#%%
with open(mention_embeddings_path) as f:
    mention_embeddings = json.load(f)
print('Loaded mention embeddings')
#%%
X_test = np.array([mention_embeddings[mention] for mention, node_id in mentions_and_node_ids])
print('Computed X_test')
#%%
node_embeddings = {}
with open(node_embeddings_path) as f:
    lines = f.readlines()[1:]
    for line in lines:
        tokens = line.split()
        node_embeddings[tokens[0]] = [float(token) for token in tokens[1:]]
print('Loaded node embeddings')
#%%
preds = normalize(model.predict(X_test))
df_node_embeddings = pd.DataFrame(node_embeddings).T
similarities = np.matmul(preds, normalize(df_node_embeddings).T)
df_sims = pd.DataFrame(similarities, columns=df_node_embeddings.index)
print('Computed df_sims')
#%%
#plt.hist(similarities)
#plt.show()
#%%
pred_names = df_sims.idxmax(axis=1)
#%%
node_ids = [node_id for mention, node_id in mentions_and_node_ids]
#%%
(pred_names == node_ids).sum() / len(node_ids)
#%%
graph = nx.read_edgelist(configs['ontobiotope_graph'])
#%%
depths = nx.shortest_path_length(graph, source='OBT:000000')
depth_of_tree = max(depths.values())
diameter = nx.algorithms.distance_measures.diameter(graph)
#depths = distances['OBT:000000']['OBT:000000']
#%%
sps = []
for pred, node_id in zip(pred_names, node_ids):
#    lcs = nx.lowest_common_ancestor(graph, pred, node_id)
#    depth_lcs = depths[lcs]
#    depth_pred = depths[pred]
#    depth_label = depths[node_id]
#    wp = 2*depth_lcs / (depth_pred + depth_label)
#    wps.append(wp)
    sp = nx.shortest_path_length(graph, source=pred, target=node_id)   
    sps.append(sp)
#%%
import seaborn as sns
sns.distplot(sps, bins=np.arange(0,max(sps)+1,1))
plt.xticks(range(-2,max(sps)+1,1))
plt.show()


