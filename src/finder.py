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

    def find_related_docs(self, graph, node_id, max_distance=1):
        distances = nx.shortest_path_length(graph, source=node_id)
        related_nodes = {node_id: dist for node_id, dist in distances.items() if dist <= max_distance}
        related_nodes = sorted(related_nodes.items(), key=lambda x: x[1])

        search_results = {}
        for node_id, dist in related_nodes:
            if node_id in self.inverted_index:
                docs_in_node = self.inverted_index[node_id]
                search_results[(dist, node_id, graph.nodes(data=True)[node_id]['name'])] = docs_in_node

        return search_results

    def display_search_results(self, search_results, doc_path):
        doc_to_nodes = defaultdict(list)
        for t, docs in search_results.items():
            for doc in docs:
                doc_to_nodes[doc].extend(t)

        display = []
        processed_results = set()
        for t, docs in search_results.items():
            for doc in docs:
                if doc not in search_results:
                    with open(doc_path + doc.replace('.a2', '.txt')) as f:
                        file_content = f.read().strip()
                    tag = ', '.join([str(node) for node in doc_to_nodes[doc]])
                    display.append(tag + '\n' + file_content)
                    processed_results.add(doc)

        with open('./data/search_results.txt', 'w') as f:
            f.write('\n\n'.join(display))

        return ('\n'.join(display))
