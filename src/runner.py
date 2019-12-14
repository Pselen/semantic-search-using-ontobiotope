import json
#%%
from src.ontobiotope import OntoBiotope
from src.utils import extract_mention_node_matchings, matching_to_embedding
from src.mention_set import MentionSet
#%%
with open('configs.json') as f:
    configs = json.load(f)
#%%
# from stellargraph.data import load_dataset_BlogCatalog3
# blog_graph = stellargraph.data.load_dataset_BlogCatalog3('C:/Users/rizao/Downloads/BlogCatalog-dataset/BlogCatalog-dataset/data')
# visualize_embeddings(configs['node_embeddings'])
#%%
ontobiotope = OntoBiotope(configs['ontobiotope_raw'], configs['train'])
ontobiotope.initialize(save_path=configs['ontobiotope_nx'])
ontobiotope.enrich_with_cooccurence(save_path=configs['ontobiotope_enriched'])
node_embeddings = ontobiotope.learn_embeddings(embedding_path=configs['node_embeddings'])
#%%
mention_files = [configs['train'], configs['dev']]
mention_set = MentionSet(mention_files)
mention_embeddings = mention_set.learn_embeddings(pretrained_word_embeddings_path=configs['word_embeddings_100'],
                                                  save_path=configs['mention_embeddings_100'])
#%%
train_matchings = extract_mention_node_matchings(configs['train'])
X_train, Y_train = matching_to_embedding(train_matchings)
