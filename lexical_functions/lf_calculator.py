# TAKEN FROM CALCULATOR, MODIFIED
import numpy as np
from pymorphy2 import MorphAnalyzer

from itertools import combinations
from collections import defaultdict
from gensim.models import KeyedVectors

from utils import load_links, load_syntagrus_lf, lintransform, get_train_vectors, most_similar_by_wv

morph = MorphAnalyzer()


TRAINING_DATA = {
    'links': {
        'attr': load_links(lf='attr', b_pos='NOUN', c_pos='ADJF')
    },
    'LF': {
        'MAGN': load_syntagrus_lf(lf='_MAGN'),
        "BON": load_syntagrus_lf(lf="_BON"),
        "ANTIBON": load_syntagrus_lf(lf="_ANTIBON")
    }
}


VECTOR_MODELS = {
    'fontanka_w2v': KeyedVectors.load_word2vec_format('../models/fontanka_word2vec.bin', binary=True),
    'nplus1_w2v': KeyedVectors.load_word2vec_format('../models/nplus1_word2vec.bin', binary=True),
    'librusec_w2v': KeyedVectors.load_word2vec_format('../models/librusec_word2vec.bin', binary=True),
    'stihi_ru_w2v': KeyedVectors.load_word2vec_format('../models/stihi_ru_word2vec.bin', binary=True)
}


for model_name, word2vec_model in VECTOR_MODELS.items():
    word2vec_model.fill_norms()


RELATION_TYPE_TO_CLASS = {
    'MAGN': 'LF',
    'BON': 'LF',
    'ANTIBON': 'LF',
    'attr': 'links'
}


RELATION_TYPE_POS = {
    'MAGN': ['NOUN', 'ADJ'],
    'BON': ['NOUN', 'ADJ'],
    'ANTIBON': ['NOUN', 'ADJ'],
    'attr': ['NOUN', 'ADJ']
}

UD_POS_CORR = {
    'NOUN': '_NOUN',
    'ADJ': '_ADJ'
}

POS_CORR = {
    'fontanka_w2v': UD_POS_CORR,
    'nplus1_w2v': UD_POS_CORR,
    'librusec_w2v': UD_POS_CORR,
    'stihi_ru_w2v': UD_POS_CORR
}


TRANSFORM_DATA = {}
for v, k in RELATION_TYPE_TO_CLASS.items():
    if not TRANSFORM_DATA.get(k):
        TRANSFORM_DATA[k] = defaultdict(dict)
    for m in VECTOR_MODELS:
        if not TRANSFORM_DATA[k][v].get(m):
            TRANSFORM_DATA[k][v][m] = {}
        TRANSFORM_DATA[k][v][m][m] = lintransform(*get_train_vectors(
            TRAINING_DATA[k][v], VECTOR_MODELS[m], VECTOR_MODELS[m],
            POS_CORR[m][RELATION_TYPE_POS[v][0]],
            POS_CORR[m][RELATION_TYPE_POS[v][1]]
        ))
    for (m, c) in combinations(VECTOR_MODELS, 2):
        TRANSFORM_DATA[k][v][m][c] = lintransform(*get_train_vectors(
            TRAINING_DATA[k][v], VECTOR_MODELS[m], VECTOR_MODELS[c],
            POS_CORR[m][RELATION_TYPE_POS[v][0]],
            POS_CORR[c][RELATION_TYPE_POS[v][1]]
        ))
        TRANSFORM_DATA[k][v][c][m] = lintransform(*get_train_vectors(
            TRAINING_DATA[k][v], VECTOR_MODELS[c], VECTOR_MODELS[m],
            POS_CORR[c][RELATION_TYPE_POS[v][0]],
            POS_CORR[m][RELATION_TYPE_POS[v][1]]
        ))


def get_collocates_for_word_type(word, pos, base_space, coll_space, relation, max_items):
    try:
        cl = RELATION_TYPE_TO_CLASS[relation]
    except KeyError:
        print('Unsupported relation type!')
    try:
        test_vec = VECTOR_MODELS[base_space][word + POS_CORR[base_space][pos]]
    except KeyError:
        print(f"Could not find base word '{word}' in the model! Check POS tag or try another one.")
    trans_vec = TRANSFORM_DATA[RELATION_TYPE_TO_CLASS[relation]][relation][base_space][coll_space].dot(test_vec.T)
    trans_vec = trans_vec / np.linalg.norm(trans_vec)
    conv_pos = POS_CORR[coll_space][RELATION_TYPE_POS[relation][1]]
    similar = most_similar_by_wv(VECTOR_MODELS[coll_space], trans_vec, topn=max_items, pos=conv_pos)
    return {'concept_list': similar[:max_items]}


if __name__ == '__main__':
    print(get_collocates_for_word_type('человек', 'NOUN', 'librusec_w2v', 'stihi_ru_w2v', 'MAGN', 20))




