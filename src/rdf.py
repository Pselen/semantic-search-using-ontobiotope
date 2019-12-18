import json
from rdflib import URIRef, BNode, Literal, Namespace
from rdflib.namespace import RDF, FOAF
from rdflib import Graph

from src.ontobiotope import  OntoBiotope
#%%
with open('configs.json') as f:
    configs = json.load(f)

ontobiotope = OntoBiotope(configs['ontobiotope_raw'])
ontobiotope.load_graph(configs['ontobiotope_enriched'])
#%%
g = Graph()

cooccur_predicate = URIRef('http://example.org/predicates/cooccur')
g.add((cooccur_predicate, FOAF.name, Literal('cooccurs')))
habitat_base = URIRef('http://example.org/habitats/')
id_to_class = {node: URIRef(habitat_base + node) for node in ontobiotope.graph.nodes()}

for edge in ontobiotope.graph.edges(data=True):
    if edge[-1]['label'] == 'cooccur':
        g.add((id_to_class[edge[0]], cooccur_predicate, id_to_class[edge[1]] ))
        g.add((id_to_class[edge[1], cooccur_predicate, id_to_class[edge[0]] ))
    else:
        g.add((id_to_class[edge[0]], RDF.type, id_to_class[edge[1]] ))

rdf = g.serialize(format='turtle').decode('utf-8')
with open('./data/habitats.ttl', 'w') as f:
    f.write(rdf)
#%%

habitats = URIRef('http://example.org/habitat')

h1 = URIRef('http://example.org/habitat/h1')
h2 = URIRef('http://example.org/habitat/h2')

g.add((h1, RDF.type, habitats))
g.add((h1, FOAF.name, Literal('OBT:000111')))
g.add((h1, cooccurs, h2))
g.add((h1, RDF.type, h2))
g.add((h2, RDF.type, habitats))
# g.add( (bob, RDF.type, FOAF.Person) )
# g.add( (bob, FOAF.name, name) )
# g.add( (bob, FOAF.knows, linda) )
# g.add( (linda, RDF.type, FOAF.Person) )
# g.add( (linda, FOAF.name, Literal('Linda') ))

# temp = str(g.serialize(format='turtle'))

#%%

