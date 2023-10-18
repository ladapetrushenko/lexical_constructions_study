import os
import pickle

from gensim.models import Word2Vec

PATH = '/home/lada/corpora/processed/'


for file in os.listdir(PATH):
    if file == 'fontanka.pickle':
        with open(PATH + file, 'rb') as pickled_sentences:
            sentences = pickle.load(pickled_sentences)
        fontanka = Word2Vec(sentences, vector_size=200, window=5, min_count=15, sorted_vocab=1, cbow_mean=1)
        fontanka.wv.save_word2vec_format('../models/fontanka_word2vec.bin', binary=True)
    if file == 'nplus1.pickle':
        with open(PATH + file, 'rb') as pickled_sentences:
            sentences = pickle.load(pickled_sentences)
        nplus1 = Word2Vec(sentences, vector_size=150, window=5, min_count=15, sorted_vocab=0, cbow_mean=1)
        nplus1.wv.save_word2vec_format('../models/nplus1_word2vec.bin', binary=True)
    if file == 'librusec.pickle':
        with open(PATH + file, 'rb') as pickled_sentences:
            sentences = pickle.load(pickled_sentences)
        librusec = Word2Vec(sentences, vector_size=100, window=5, min_count=15, sorted_vocab=0, cbow_mean=1)
        librusec.wv.save_word2vec_format('../models/librusec_word2vec.bin', binary=True)
    if file == 'stihi_ru.pickle':
        with open(PATH + file, 'rb') as pickled_sentences:
            sentences = pickle.load(pickled_sentences)
        stihi_ru = Word2Vec(sentences, vector_size=150, window=5, min_count=10, sorted_vocab=1, cbow_mean=1)
        stihi_ru.wv.save_word2vec_format('../models/stihi_ru_word2vec.bin', binary=True)