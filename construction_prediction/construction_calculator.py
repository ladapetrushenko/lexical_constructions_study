# TAKEN FROM Server/calculators/W2V, MODIFIED
import os
import pickle

from gensim.models import Word2Vec, KeyedVectors
from pymorphy2 import MorphAnalyzer

morph = MorphAnalyzer()

PATH = '/home/lada/corpora/processed/'


def filter_results(model, word, target_pos, collocate_pos, topn, restrict_vocab):
    collocates = []
    target_word = '_'.join((word, target_pos))
    for coll, similarity in model.similar_by_word(target_word, topn=topn, restrict_vocab=restrict_vocab):
        try:
            coll_word, pos = coll.split('_')
            if pos == collocate_pos:
                collocates.append((coll_word, similarity))
            if len(collocates) == topn:
                break
        except ValueError:
            continue
    return collocates


def get_collocates_for_word_type(model, word, target_pos, topn, restrict_vocab):
    collocate_pos = 'NOUN' if target_pos == 'ADJ' else 'ADJ'

    print('Loading collocations...')
    collocates = filter_results(model, word, target_pos, collocate_pos, topn * 100, restrict_vocab)
    for collocate_with_score in collocates[:topn]:
        collocate = collocate_with_score[0]
        similarity_score = round(collocate_with_score[1], 3)
        noun = word if target_pos == 'NOUN' else collocate
        adj = word if target_pos == 'ADJ' else collocate
        try:
            # Чтобы была конструкция, в которой один элемент склоняется
            adj = morph.parse(adj)[0].inflect({morph.parse(noun)[0].tag.gender}).word
            # Чтобы исключить результаты типа 'человечный человек'
            if not adj[:3] == noun[:3]:
                # if noun == 'день' and not 'днев' in adj and not 'недел' in adj:
                print(f'\t{adj} {noun}: {similarity_score}')
        except AttributeError:
            continue


if __name__ == '__main__':
    nplus1 = KeyedVectors.load_word2vec_format('../models/nplus1_word2vec.bin', binary=True)
    fontanka = KeyedVectors.load_word2vec_format('../models/fontanka_word2vec.bin', binary=True)
    librusec = KeyedVectors.load_word2vec_format('../models/librusec_word2vec.bin', binary=True)
    stihi_ru = KeyedVectors.load_word2vec_format('../models/stihi_ru_word2vec.bin', binary=True)