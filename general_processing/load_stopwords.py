# FINAL
from string import punctuation, digits
from nltk.corpus import stopwords


def load_stopwords():
    nltk_stopwords = stopwords.words('russian')
    yandex_wordstat = '../data/wordstat_stopwords.txt'
    with open(yandex_wordstat) as stopwords_path:
        yandex_wordstat = stopwords_path.read().splitlines()
    chars = list(punctuation) + ['--'] + list(digits)
    united_stopwords = set(nltk_stopwords + yandex_wordstat + chars)
    return united_stopwords