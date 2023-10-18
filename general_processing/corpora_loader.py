import os
import pickle
import pymorphy2

from datasets import load_dataset
from load_stopwords import load_stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# PATHS
FONTANKA_PATH = '/home/lada/corpora/Fontanka/texts/'
NPLUS1_PATH = '/home/lada/corpora/nplus1/texts/'
PATH_TO_WRITE_TO = '/home/lada/corpora/processed/'

morph = pymorphy2.MorphAnalyzer()


class CorpusProcessor:

    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        self.stopwords = load_stopwords()
        self.sentences = []  # Representing the corpus as a list of lists of tokens for training (all tokens)
        self.tokens_count = 0
        self.sentences_count = 0

    def clean_data(self, text):
        for sent in sent_tokenize(text):
            sent_processed = []
            for token in word_tokenize(sent):
                if token not in self.stopwords and len(token) > 2:
                    parse_result = morph.parse(token)[0]
                    lemma, pos = parse_result.normal_form, parse_result.tag.POS
                    # Меняем теги 'ADJF' и 'ADJS' на 'ADJ'
                    if pos == 'ADJF' or pos == 'ADJS':
                        pos = pos[:-1]
                    sent_processed.append(f'{lemma}_{pos}')
            if len(sent_processed) > 2:
                self.tokens_count += len(sent_processed)
                self.sentences_count += 1
                self.sentences.append(sent_processed)

    def standard_processor(self):
        for file in os.listdir(self.corpus_path):
            with open(self.corpus_path + file, 'r') as file_to_read:
                text = str(file_to_read.read().lower())
                self.clean_data(text)

    def hf_dataset_processor(self, limit):
        corpus = load_dataset(self.corpus_path, split='train', streaming=True)
        texts = []
        for text in corpus:
            if len(text) < 999000:
                texts.append(text['text'])
                if len(texts) >= limit:
                    break
        for text in texts:
            self.clean_data(text)

    def save_pickle(self, pickle_save_path):
        with open(pickle_save_path, 'wb') as pickle_path:
            pickle.dump(self.sentences, pickle_path)


if __name__ == '__main__':
    pass
    # В Фонтанке нужно еще объединить корпуса в один (переместить их в одну папку)
    # for folder in os.listdir(FONTANKA_PATH):
    #     for file in os.listdir(f'{FONTANKA_PATH}/{folder}'):
    #         os.replace(f'{FONTANKA_PATH}/{folder}/{file}', f'{FONTANKA_PATH}/{file}')
    # for folder in (2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017):
    #     os.rmdir(f'{FONTANKA_PATH}/{folder}')

    # fontanka = CorpusProcessor(FONTANKA_PATH)
    # fontanka.standard_processor()
    # fontanka.save_pickle(PATH_TO_WRITE_TO + 'fontaka.pickle')
    # print(fontanka.tokens_count, fontanka.sentences_count)

    # nplus1 = CorpusProcessor(NPLUS1_PATH)
    # nplus1.standard_processor()
    # nplus1.save_pickle(PATH_TO_WRITE_TO + 'nplus1.pickle')
    # print(nplus1.sentences)
    # print(nplus1.tokens_count, nplus1.sentences_count)

    # stihi_ru = CorpusProcessor('IlyaGusev/stihi_ru')
    # stihi_ru.hf_dataset_processor(100000)
    # stihi_ru.save_pickle(PATH_TO_WRITE_TO + 'stihi_ru.pickle')
    # print(stihi_ru.tokens_count, stihi_ru.sentences_count)

    # proza_ru = CorpusProcessor('IlyaGusev/librusec')
    # proza_ru.hf_dataset_processor(1000)
    # proza_ru.save_pickle(PATH_TO_WRITE_TO + 'proza_ru.pickle')
    # print(proza_ru.tokens_count, proza_ru.sentences_count)
