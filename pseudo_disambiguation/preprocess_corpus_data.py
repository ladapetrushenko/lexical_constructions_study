import os
import pickle


filenames = ['../data/collocates_for_ADJ.txt', '../data/collocates_for_NOUN.txt']
with open(filenames[0], 'r') as txt_for_collocates_1, open(filenames[1], 'r') as txt_for_collocates_2:
    test_set = txt_for_collocates_1.read()
    test_set += txt_for_collocates_2.read()

# Choosing right bigrams to delete the sentences with these bigrams from the texts
right_bigrams = []
for line in test_set.splitlines():
    right_bigram = line.split('\t')[0] + ' ' + line.split('\t')[2]
    right_bigrams.append(right_bigram)
print(right_bigrams)

for collection in os.listdir('../corpora'):
    if collection == 'fontanka.pickle':
        with open(f'../corpora/{collection}', 'rb') as pickled_dataset:
            dataset = pickle.load(pickled_dataset)
        sentences_to_delete = []
        for sentence in dataset:
            for right_bigram in right_bigrams:
                if right_bigram in ' '.join(sentence):
                    try:
                        print(sentence)
                        dataset.remove(sentence)
                    except ValueError:
                        continue
        print(len(dataset))
        with open(f'../corpora/processed_{collection}', 'wb') as collection_file:
            pickle.dump(dataset, collection_file)