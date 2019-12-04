import json
import os
import re
import networkx as nx
#%%
def parse_ontotbiotope(ontology_path):    
    with open(ontology_path) as f:
        # obtain each node by discarding initial comments
        nodes = f.read().split('\n\n')[2:]
    
    # ignore first lines that contains '[Term]'    
#    features = [node[node.index('\n')+1:].strip() for node in nodes]
#    feature_names = list(set([line[:line.index(':')] for line in '\n'.join(features).strip().split('\n')]))
    
    graph = nx.Graph()
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
                    graph.add_edge(node_id, value[:value.index('!')].strip())
                else:
                    graph.add_edge(node_id, value.strip())
            
        graph.nodes[node_id]['name'] = name
        graph.nodes[node_id]['synonyms'] = '\n'.join(synonyms)
    
    return graph

def enrich_ontobiotope_with_cooccurence(graph, training_docs_path, window_size=3):
    counter = 0
    a2_files = [f for f in os.listdir(training_docs_path) if f.endswith('.a2')]
    for file in a2_files:
        path = training_docs_path + file
        with open(path) as f:
            annotations = [l.split() for l in f.readlines()]
        
        node_ids = [annotation[3].split(':')[-1] for annotation in annotations 
                    if annotation[1] == 'OntoBiotope']
        for idx,node_id in enumerate(node_ids):
            for i in range(1,window_size+1):
                if idx+i < len(node_ids): 
                    neighbor_id = node_ids[idx+i]
                    if neighbor_id != node_id:
                        counter = counter + 1
                        graph.add_edge('OBT:' + node_id, 'OBT:' + neighbor_id)
            
    return graph
#%%
with open('configs.json') as f:
    configs = json.load(f)
#%%
ontology_path = configs['ontobiotope']
training_docs_path = configs['train']
window_size = 37 # window size that edges all edges
#%%
graph = parse_ontotbiotope(ontology_path)
print(f'Nodes:{graph.number_of_nodes()}, Edges:{graph.number_of_edges()}')
enriched_graph = enrich_ontobiotope_with_cooccurence(graph, training_docs_path, window_size)
print(f'Nodes:{graph.number_of_nodes()}, Edges:{graph.number_of_edges()}')
#%%
nx.readwrite.write_edgelist(graph, configs['ontobiotope_graph'], comments=None, data=False)

#%%
nx.readwrite.write_edgelist(enriched_graph, configs['enriched_ontobiotope'], comments=None, data=False)
#%%
# >python -m openne --method node2vec 
# --input ../../data/enriched_ontobiotope.gph 
# --graph-format edgelist --output ../../data/node_embeddings.txt 
# --q 0.25 --p 0.25 --representation-size=100