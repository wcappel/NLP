import nltk
import ssl
from nltk.corpus import opinion_lexicon
from nltk.stem.porter import PorterStemmer
import math
import numpy

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

# lrTraining = []
# for x in testData:
#     newList = []
#     for key in x[0]:
#         newList.append(key)
#     bigList = [newList, x[1]]
#     lrTraining.append(bigList)
#
# print(lrTraining)

porter = PorterStemmer()

def lexStemmer(lexicon):
    stemmedLexicon = []
    for word in lexicon:
        stemmedLexicon.append(porter.stem(word))
    return stemmedLexicon

nltkPosLex = opinion_lexicon.positive()
nltkNegLex = opinion_lexicon.negative()
posLex = ["".join(list_of_words) for list_of_words in nltkPosLex]
negLex = ["".join(list_of_words) for list_of_words in nltkNegLex]

stemmedPosLex = lexStemmer(posLex)
stemmedNegLex = lexStemmer(negLex)

def featureCount(review):
    frequencies = [0, 0, 0, 0, 0, 0]
    for word in review[0]:
        if word in stemmedPosLex:
            frequencies[0] += 1
        elif word in stemmedNegLex:
            frequencies[1] += 1
    restrung = " ".join(review[0])
    reviewBigrams = list(nltk.bigrams(restrung.split()))
    #print(*map(' '.join, reviewBigrams), sep=', ')
    for bigram in reviewBigrams:
        print(bigram)
        if bigram[0] == ('not') and bigram[1] == 'good':
            frequencies[2] += 1
        elif bigram[0] == ('i') and bigram[1] == 'like':
            frequencies[3] += 1
        elif bigram[0] == ('not') and bigram[1] == 'bad':
            frequencies[4] += 1
        elif bigram[0] == ('dont') and bigram[1] == 'like':
            frequencies[5] += 1
    return frequencies

print(featureCount((['i', 'like'], 'pos')))