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

    # def find_related_docs(self, graph, node_id, max_distance=1):
    #     sps = nx.shortest_path_length(graph, source=node_id)
    #     # close_nodes = [node for node, dist in sps.items() if dist <= max_distance]
    #     dist_to_nodes = defaultdict(list)
    #     for node_id, dist in sps.items():
    #         dist_to_nodes[dist].append(node_id)

    #     dist_to_docs = defaultdict(list)
    #     for dist, node_id in dist_to_nodes rdf.items():
    #         if node_id in self.inverted_index:
    #             dist_to_docs[dist].extend(self.inverted_index[node_id])


    #     close_docs = [set(self.inverted_index[node]) for node in close_nodes if node in self.inverted_index]
    #     unique_docs = set.union(*close_docs)
    #     close_docs_by_dist = [(sps[doc], doc) for doc in unique_docs]
    #     close_docs_by_dist.sort()
    #     return close_docs_by_dist
