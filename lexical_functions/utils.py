# TAKEN FROM CALCULATOR
# MODIFIED
import numpy as np
from pymorphy2 import MorphAnalyzer


pm_corr = {
    "_A": "ADJF",
    "_S": "NOUN",
    "_ADJ": "ADJF",
    "_NOUN": "NOUN"
}

morph = MorphAnalyzer()


def load_links(lf='attr', b_pos='NOUN', c_pos='ADJF'):
    coll_ = []
    with open(f'../data/synt_links_{lf}') as f:
        for line in f:
            line = line.rstrip('\r\n').split('\t')
            if c_pos in [x.tag.POS for x in morph.parse(line[0])] \
                    and b_pos in [x.tag.POS for x in morph.parse(line[1])]:
                pair = (line[1].lower(), line[0].lower())
                if pair not in coll_:
                    coll_.append(pair)
    return coll_


def load_syntagrus_lf(lf='MAGN'):
    coll_lf = []
    with open('../data/lf.SynTagRus.tsv') as f:
        for line in f:
            line = line.rstrip('\r\n').split('\t')
            if line[0] == lf:
                pair = (line[1].lower(), line[2].lower())
                if pair not in coll_lf:
                    coll_lf.append(pair)
    return coll_lf


def lintransform(M_train, C_train):
    return np.linalg.pinv(M_train).dot(C_train).T


def get_train_vectors(trainset, bspace, cspace, b_pos, c_pos):
    B_train, C_train = [], []
    for base, collocate in trainset:
        try:
            b_vec = bspace[base + b_pos]
            c_vec = cspace[collocate + c_pos]
            B_train.append(b_vec)
            C_train.append(c_vec)
        except KeyError:
            pass
    return np.array(B_train), np.array(C_train)


def similar_by_vector(self, vectenter, topn=10, restrict_vocab=20000):
    self.fill_norms()
    dists = np.dot(self.get_normed_vectors(), vectenter)
    best = np.argsort(dists)[::-1][:topn*10]
    return [(self.index_to_key[sim], float(dists[sim])) for sim in best if sim < restrict_vocab]


def most_similar_by_wv(self, vectenter, topn=5, pos='_V'):
    best = similar_by_vector(self, vectenter, topn=topn*5)
    return [(word.split("_")[0], score) for (word, score) in filter(lambda x: x[0].endswith(pos), best)][:topn]




