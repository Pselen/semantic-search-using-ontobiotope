import os
import re

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

from gensim.models import KeyedVectors
#%%


def visualize_embeddings(embeddings_path):
    print('Loading embeddings...')
    keyed_vectors = KeyedVectors.load(embeddings_path, mmap='r')
    node_embeddings = keyed_vectors.wv.vectors

    tsne = TSNE(n_components=2)
    node_embeddings_2d = tsne.fit_transform(node_embeddings)

    plt.figure(figsize=(7, 7))
    plt.axes().set(aspect="equal")
    plt.scatter(node_embeddings_2d[:, 0],
                node_embeddings_2d[:, 1],
                cmap="jet", alpha=0.7)
    plt.title('Visualization of node embeddings by TSNE')
    plt.show()


def normalize_mention(mention):
    return re.sub(r'[,.;@#?!&$-:/]+\ *', ' ', mention).strip()


def extract_mention_node_matchings(path):
    a1_files = [f for f in os.listdir(path) if f.endswith('.a1')]
    #mention_to_node_id = {}
    mentions_and_node_ids = []
    for a1_file in a1_files:
        with open(path + a1_file, encoding='latin5') as a1:
            annotations = [l.split('\t') for l in a1.readlines()]
            tag_to_mention = {annotation[0]: normalize_mention(annotation[-1]) for annotation in annotations if 'Habitat' in annotation[1]}

        a2_file = a1_file.replace('.a1', '.a2')
        with open(path + a2_file) as a2:
            annotations = [l.split() for l in a2.readlines()]

        tag_to_node_id = {annotation[2].split(':')[-1]: 'OBT:' + annotation[3].split(':')[-1] for annotation in annotations if annotation[1] == 'OntoBiotope'}

        for tag, mention in tag_to_mention.items():
            node_id = tag_to_node_id[tag]
            mentions_and_node_ids.append((mention, node_id))

    return list(set(mentions_and_node_ids))


def matching_to_embedding(matching, mention_embeddings, node_embeddings):
    X, Y = [], []
    for mention, node_id in matching:
        X.append(mention_embeddings[mention])
        Y.append(node_embeddings[node_id])

    X = normalize(X)
    Y = normalize(Y)
    return X, Y
