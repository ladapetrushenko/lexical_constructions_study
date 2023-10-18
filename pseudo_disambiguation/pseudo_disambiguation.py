# Final
import numpy as np

from numpy import linalg
from scipy.spatial import distance
from gensim.models import Word2Vec, FastText


class PseudoDisambiguation:

    def __init__(self, training_set, test_set, parameters):
        self.training_set = training_set
        self.test_set = test_set
        self.parameters = parameters
        self.metric = parameters[0]
        self.vector_size = parameters[1]
        self.window = parameters[2]
        self.min_count = parameters[3]
        self.cbow_mean = parameters[4]
        self.sorted_vocab = parameters[5]
        self.max_vocab = parameters[6]
        self.model = None  # Здесь будет актуальная модель, обученная с параметрами из self.parameters

    @staticmethod
    def euclidean_distance(vector_1, vector_2):
        return linalg.norm(vector_1 - vector_2)

    @staticmethod
    def squared_euclidean_distance(vector_1, vector_2):
        return distance.sqeuclidean(vector_1, vector_2)

    @staticmethod
    def cosine_distance(vector_1, vector_2):
        return np.dot(vector_1, vector_2) / (linalg.norm(vector_1) * linalg.norm(vector_2))

    @staticmethod
    def correlation_distance(vector_1, vector_2):
        return distance.correlation(vector_1, vector_2)

    def calculate_similarity(self, vector_1, vector_2):
        if self.metric == 'euclidean':
            return round(self.euclidean_distance(vector_1, vector_2), 2)
        elif self.metric == 'sqeuclidean':
            return round(self.squared_euclidean_distance(vector_1, vector_2), 2)
        elif self.metric == 'cosine':
            return round(self.cosine_distance(vector_1, vector_2), 2)
        elif self.metric == 'correlation':
            return round(self.correlation_distance(vector_1, vector_2), 2)

    def predict(self, target_word, collocate_1, collocate_2):
        try:
            collocate_1_similarity = self.calculate_similarity(self.model.wv[target_word], self.model.wv[collocate_1])
            collocate_2_similarity = self.calculate_similarity(self.model.wv[target_word], self.model.wv[collocate_2])
            if collocate_1_similarity > collocate_2_similarity:
                return collocate_1
            elif collocate_2_similarity > collocate_1_similarity:
                return collocate_2
            else:
                return f'{collocate_1} | {collocate_2}'  # Если равны
        except KeyError:
            return collocate_1

    def evaluate(self):
        tp_counts, fp_counts, fn_counts = 0, 0, 0
        for row in self.test_set.splitlines():
            row = row.split('\t')
            right_collocate, bad_collocate, target = row[0], row[1], row[2]
            predicted_result = self.predict(target, right_collocate, bad_collocate)
            if predicted_result == right_collocate:
                tp_counts += 1
            elif predicted_result != right_collocate:
                if predicted_result == f'{right_collocate} | {bad_collocate}':
                    fn_counts += 1
                fp_counts += 1
        precision = tp_counts / (tp_counts + fp_counts) * 100
        recall = tp_counts / (tp_counts + fn_counts) * 100
        try:
            f_score = (2 * (precision * recall)) / (precision + recall)
        except ZeroDivisionError:
            f_score = precision
        print(f'Parameters: {self.parameters}\n'
              f'F-score: {round(f_score, 2)}%\n'
              f'Precision: {round(precision, 2)}%\nRecall: {round(recall, 2)}%\n'
              f'==========')
        return self.parameters, round(f_score, 2), round(precision, 2), round(recall, 2)


class TrainWord2Vec(PseudoDisambiguation):

    def __init__(self, training_set, test_set, parameters):
        super().__init__(training_set, test_set, parameters)

    def train_word2vec(self):
        self.model = Word2Vec(sentences=self.training_set, vector_size=self.vector_size, window=self.window,
                              min_count=self.min_count, cbow_mean=self.cbow_mean, sorted_vocab=self.sorted_vocab,
                              workers=-1, max_vocab_size=self.max_vocab)


class TrainFastText(PseudoDisambiguation):

    def __init__(self, training_set, test_set, parameters):
        super().__init__(training_set, test_set, parameters)

    def train_fasttext(self):
        self.model = FastText(sentences=self.training_set, vector_size=self.vector_size, window=self.window,
                              min_count=self.min_count, cbow_mean=self.cbow_mean, sorted_vocab=self.sorted_vocab,
                              workers=-1, max_vocab_size=self.max_vocab)
