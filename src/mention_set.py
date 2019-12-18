import os
import re
import json

import numpy as np

from sklearn import preprocessing


class MentionSet:
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.mentions = []
        self._populate_habitat_mentions()

    def _normalize_mention(self, mention):
        return re.sub(r'[,.;@#?!&$-:/]+\ *', ' ', mention).strip()

    def _populate_habitat_mentions(self):
        print('Extracting mentions')
        habitat_mentions = []
        for path in self.file_paths:
            a1_files = [f for f in os.listdir(path) if f.endswith('.a1')]
            # Filter a1 files by Habitats
            for a1_file in a1_files:
                with open(path + a1_file, encoding='latin5') as a1:
                    annotations = [l.split('\t') for l in a1.readlines()]
                    habitat_annotations = [annotation[-1] for annotation in annotations if 'Habitat' in annotation[1]]
                    habitat_mentions.extend(habitat_annotations)

        # normalize mentions and get unique mentions only
        self.mentions = list(set([self._normalize_mention(mention) for mention in habitat_mentions]))

    @staticmethod
    def mention_to_embedding(mention, pretrained_word_embeddings):
        word_vector_shape = len(list(pretrained_word_embeddings.values())[0])
        mention_embedding = np.zeros((word_vector_shape))
        word_count_in_mention = 0
        for word in mention.split():
            word = word.strip()
            if word in pretrained_word_embeddings:
                mention_embedding = mention_embedding + pretrained_word_embeddings[word]
                word_count_in_mention = word_count_in_mention + 1
            elif word.lower() in pretrained_word_embeddings:
                mention_embedding = mention_embedding + pretrained_word_embeddings[word.lower()]
                word_count_in_mention = word_count_in_mention + 1

        if word_count_in_mention > 0:
            mention_embedding = mention_embedding / word_count_in_mention
            return preprocessing.normalize([mention_embedding], norm='l2').tolist()

        return np.zeros((word_vector_shape))[0].tolist()

    def learn_embeddings(self, pretrained_word_embeddings):
        return {mention: MentionSet.mention_to_embedding(mention, pretrained_word_embeddings) for mention in self.mentions}

    @staticmethod
    def save_embeddings(mention_embeddings, embeddings_path):
        with open(embeddings_path, 'w') as f:
            json.dump(mention_embeddings, f)

    @staticmethod
    def load_embeddings(mention_embeddings_path):
        with open(mention_embeddings_path) as f:
            return json.load(f)
