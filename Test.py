import nltk
import ssl
from nltk.corpus import opinion_lexicon
from nltk.stem.porter import PorterStemmer

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('opinion_lexicon')
# print(289*742)

# str = "What's a good sentence."
# bigram = list(nltk.bigrams(str.split()))
# print(*map(' '.join, bigram), sep=', ')



print(stemmedPosLex)