import os
import re

import networkx as nx

from stellargraph.data import BiasedRandomWalk
from stellargraph import StellarGraph
from gensim.models import Word2Vec, KeyedVectors
#%%


class OntoBiotope:

    def __init__(self, ontobiotope_path, abstract_path=None):
        self.ontobiotope_path = ontobiotope_path
        self.abstract_path = abstract_path
        self.graph = nx.Graph()

    def initialize(self, save_path=None):
        with open(self.ontobiotope_path) as f:
            # obtain each node by discarding initial comments
            nodes = f.read().split('\n\n')[2:]

        for node in nodes:
            # Ignore '[Term]'s
            lines = node.strip().split('\n')[1:]
            for line in lines:
                synonyms = []
                sep = line.index(':')
                f_name, value = line[:sep], line[sep+1:]
                if f_name == 'id':
                    node_id = value.strip()
                elif f_name == 'name':
                    name = value.strip()
                elif f_name == 'synonym':
                    if '"' in value:
                        synonyms.append(re.search('(\".*\")', value).group(0))
                    else:
                        synonyms.append(value.strip())
                elif f_name == 'is_a':
                    if '!' in value:
                        self.graph.add_edge(node_id, value[:value.index('!')].strip(), label='taxonomy')
                    else:
                        self.graph.add_edge(node_id, value.strip(), label='taxonomy')

            self.graph.nodes[node_id]['name'] = name
            self.graph.nodes[node_id]['synonyms'] = '\n'.join(synonyms)

        if save_path:
            nx.readwrite.write_edgelist(self.graph, save_path)

    def enrich_with_cooccurence(self, window_size=37, save_path=None):
        a2_files = [f for f in os.listdir(self.abstract_path) if f.endswith('.a2')]
        for file in a2_files:
            path = self.abstract_path + file
            with open(path) as f:
                annotations = [l.split() for l in f.readlines()]

            node_ids = [annotation[3].split(':')[-1] for annotation in annotations
                        if annotation[1] == 'OntoBiotope']
            for idx, node_id in enumerate(node_ids):
                for i in range(1, window_size+1):
                    if idx+i < len(node_ids):
                        neighbor_id = node_ids[idx+i]
                        if neighbor_id != node_id:
                            self.graph.add_edge('OBT:' + node_id, 'OBT:' + neighbor_id, label='cooccur')

        if save_path:
            nx.readwrite.write_edgelist(self.graph, save_path)

    def learn_embeddings(self, embedding_dim=100, window_size=5, embedding_path=None,
              max_rw_len=50, walks_per_node=20, p=0.5, q=2.0):
        print('Running node2vec...')
        rw = BiasedRandomWalk(StellarGraph(self.graph))
        walks = rw.run(nodes=list(self.graph), length=max_rw_len, n=walks_per_node, p=p, q=q)
        print(f'Number of random walks: {len(walks)}')

        print('Running word2vec...')
        model = Word2Vec(walks, size=embedding_dim, window=window_size, min_count=0, sg=1, workers=2, iter=1)
        model.init_sims(replace=True)

        print('Saving Embeddings')
        if embedding_path:
            model.wv.save(embedding_path)

        return model.wv

    def load_embeddings(self, embeddings_path):
        return KeyedVectors.load(embeddings_path, mmap='r')
