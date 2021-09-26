import nltk
import ssl
from nltk.corpus import opinion_lexicon
from nltk.stem.porter import PorterStemmer

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# nltk.download('opinion_lexicon')
# print(289*742)

# str = "What's a good sentence."
# bigram = list(nltk.bigrams(str.split()))
# print(*map(' '.join, bigram), sep=', ')

#print(stemmedPosLex)

testData = [({'This': False, 'is': False,'a': False, 'sentence':False}, 'pos'), ({'Another': False, 'sentence': False}, 'neg')]
# lrTraining = [word for x[0], x[1]) for x in testData]

lrTraining = []
for x in testData:
    newList = []
    for key in x[0]:
        newList.append(key)
    bigList = [newList, x[1]]
    lrTraining.append(bigList)

print(lrTraining)