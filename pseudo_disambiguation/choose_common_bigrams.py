import pickle
import random
from collections import Counter

random.seed(43)


class CollocatesFinder:

    def __init__(self, target_pos):
        self.target_pos = target_pos
        self.collocate_pos = 'ADJ' if target_pos == 'NOUN' else 'NOUN'
        self.all_bigrams = []  # All bigrams in all corpora
        self.common_bigrams = None  # Common bigrams among all corpora
        self.all_collocates_freq = []  # Frequencies for all words of collocate POS in all corpora
        self.collocates_and_targets = []

    @staticmethod
    def corpus_unpickling(corpus_path):
        with open(corpus_path, 'rb') as pickled_corpus:
            corpus = pickle.load(pickled_corpus)
        return corpus

    def find_individual_bigrams(self, corpus_path):
        corpus = self.corpus_unpickling(corpus_path)
        tokens = [token for sentence in corpus for token in sentence]
        collocates_freq = Counter(list(filter(lambda token: token.endswith(self.collocate_pos), tokens)))
        # Calculating relative frequencies
        collocates_freq = {collocate: round(collocates_freq.get(collocate) / sum(collocates_freq.values()), 7)
                           for collocate in collocates_freq.keys()}

        bigrams = []
        if self.collocate_pos == 'ADJ':
            for previous, current in zip(tokens, tokens[1:]):
                if previous.endswith(self.collocate_pos) and current.endswith(self.target_pos):
                    bigrams.append(f'{previous} {current}')
        elif self.collocate_pos == 'NOUN':
            for previous, current in zip(tokens, tokens[1:]):
                if previous.endswith(self.target_pos) and current.endswith(self.collocate_pos):
                    bigrams.append(f'{previous} {current}')

        bigrams = Counter(bigrams)
        bigrams = [bigram for bigram, freq in bigrams.items() if freq >= 8]

        self.all_bigrams.append(bigrams)
        self.all_collocates_freq.append(collocates_freq)

    def find_common_bigrams(self):
        self.common_bigrams = set(self.all_bigrams[0])
        for bigrams in self.all_bigrams[1:]:
            self.common_bigrams.intersection_update(bigrams)

    def find_bad_pair_for_targets(self):
        target_part = 1 if self.target_pos == 'NOUN' else 0
        collocate_part = 0 if self.collocate_pos == 'ADJ' else 1
        collocates = [bigram.split()[collocate_part] for bigram in self.common_bigrams]
        targets = [bigram.split()[target_part] for bigram in self.common_bigrams]

        for collocate, target in zip(collocates, targets):
            all_freqs_for_this_collocate = []
            for collocates_freq in self.all_collocates_freq:
                all_freqs_for_this_collocate.append(collocates_freq.get(collocate))
            max_freq = max(all_freqs_for_this_collocate)

            all_candidates = []
            for collocates_freq in self.all_collocates_freq:
                collocates_freq = {k: v for k, v in collocates_freq.items() if v <= max_freq}
                all_candidates.append(list(collocates_freq.keys()))

            final_bad_candidates = set(all_candidates[0])
            for bigrams in all_candidates[1:]:
                final_bad_candidates.intersection_update(bigrams)

            bad_collocate = random.choice(list(final_bad_candidates))
            self.collocates_and_targets.append(f'{collocate}\t{bad_collocate}\t{target}')

    def write_to_txt(self):
        txt_for_collocates = open(f'../data/collocates_for_{self.target_pos}_1.txt', 'w')
        for combination in self.collocates_and_targets:
            txt_for_collocates.write(combination)
            txt_for_collocates.write('\n')
        txt_for_collocates.close()


if __name__ == '__main__':
    finder = CollocatesFinder('ADJ')

    collections_path = '../corpora'
    finder.find_individual_bigrams(f'{collections_path}/fontanka.pickle')
    finder.find_individual_bigrams(f'{collections_path}/nplus1.pickle')
    finder.find_individual_bigrams(f'{collections_path}/proza_ru.pickle')
    finder.find_individual_bigrams(f'{collections_path}/stihi_ru.pickle')
    finder.find_common_bigrams()
    finder.find_bad_pair_for_targets()
    finder.write_to_txt()