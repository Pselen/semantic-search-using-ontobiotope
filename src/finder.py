import os
from collections import defaultdict
import networkx as nx


class Finder:

    def __init__(self):
        self.inverted_index = {}

    def construct_inverted_index(self, docs_path):
        inverted_index = defaultdict(list)
        a2_files = [f for f in os.listdir(docs_path) if f.endswith('.a2')]
        # mentions_and_node_ids = []
        for a2_file in a2_files:
            with open(docs_path + a2_file) as a2:
                annotations = [l.split() for l in a2.readlines()]

            node_ids = set(['OBT:' + annotation[3].split(':')[-1] for annotation in annotations if annotation[1] == 'OntoBiotope'])

            for node_id in node_ids:
                inverted_index[node_id].append(a2_file)

        self.inverted_index = dict(inverted_index)

    def find_related_docs(graph, node_id, max_distance=1):
        sps = nx.shortest_path_length(graph, source=node_id)

        pass

