# FINAL
import os
import pickle
import itertools

from collections import OrderedDict
from pseudo_disambiguation import TrainWord2Vec, TrainFastText

# Все возможные параметры модели, с которыми будем обучать модель Word2Vec
METRIC = ['cosine' 'correlation','euclidean', 'sqeuclidean']
VECTOR_SIZE = [100, 150, 200, 250, 300]
WINDOW_SIZE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
MIN_COUNT = [5, 10, 15]
CBOW_MEAN = [0, 1]
SORTED_VOCAB = [0, 1]
MAX_VOCAB = [None, 30000, 60000]

parameter_combinations = list(itertools.product(METRIC, VECTOR_SIZE, WINDOW_SIZE, MIN_COUNT, CBOW_MEAN, SORTED_VOCAB,
                                                MAX_VOCAB))
print(f'Все возможные комбинации параметров: {len(parameter_combinations)}')

PATH_TO_WRITE_TO = '/home/lada/corpora/processed/'


def choose_best_parameters(collection_name: str, model_name: str, all_predictions: dict):
    output_file = open('../data/predictions_new.txt', 'a')
    output_file.write(f'{collection_name}\n{model_name}')
    best_parameters = OrderedDict(sorted(all_predictions.items(), key=lambda x: x[1], reverse=True))
    output_file.write('\nBEST PARAMETER SETS:\n')
    print('BEST PARAMETER SETS:')
    for parameters, metrics in best_parameters.items():
        output_file.write(f'Parameters: {parameters}; Precision: {metrics[1]}%; '
                          f'Recall: {metrics[2]}%; F-score: {metrics[0]}%\n')
        print(f'Parameters: {parameters}; Precision: {metrics[1]}%; '
              f'Recall: {metrics[2]}%; F-score: {metrics[0]}%')
    print()
    output_file.write('=====================\n')
    output_file.close()


if __name__ == '__main__':
    # Loading test set
    filenames = ['../data/collocates_for_ADJ.txt', '../data/collocates_for_NOUN.txt']
    with open(filenames[0], 'r') as txt_for_collocates_1, open(filenames[1], 'r') as txt_for_collocates_2:
        test_set = txt_for_collocates_1.read()
        test_set += txt_for_collocates_2.read()

    for collection in os.listdir(PATH_TO_WRITE_TO):
        if collection.endswith('.pickle'):
            with open(PATH_TO_WRITE_TO + collection, 'rb') as pickled_dataset:
                dataset = pickle.load(pickled_dataset)
            # =======================================================
            # Word2Vec
            all_w2v_predictions = {}
            for parameters in parameter_combinations:
                model = TrainWord2Vec(dataset, test_set, parameters)
                model.train_word2vec()
                predictions = model.evaluate()
                all_w2v_predictions[parameters] = list(predictions[1:])
            choose_best_parameters(collection, 'Word2Vec', all_w2v_predictions)
            # =======================================================
            # FastText
            # all_ft_predictions = {}
            # for parameters in parameter_combinations:
            #     model = TrainFastText(dataset, test_set, parameters)
            #     model.train_fasttext()
            #     predictions = model.evaluate()
            #     all_ft_predictions[parameters] = list(predictions[1:])
            # choose_best_parameters(collection, 'FastText', all_ft_predictions)