import nltk
import ssl
from nltk.corpus import opinion_lexicon
from nltk.stem.porter import PorterStemmer
import math
import numpy
import sklearn
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import pandas
import matplotlib.pyplot as plt
import random
# list1 = ['haha', 'yes', 'no', 'maybe', 'so']
# list2 = ['haha', 'yes', 'no', 'maybe', 'so']
# seed = random.randint(0, 2147483647)
# random.seed(seed)
# random.shuffle(list1)
# random.seed(seed)
# random.shuffle(list2)
# print(list1)
# print(list2)

def lexStemmer(lexicon):
    stemmedLexicon = []
    for word in lexicon:
        stemmedLexicon.append(porter.stem(word))
    return stemmedLexicon

nltkPosLex = opinion_lexicon.positive()
nltkNegLex = opinion_lexicon.negative()
posLex = ["".join(list_of_words) for list_of_words in nltkPosLex]
negLex = ["".join(list_of_words) for list_of_words in nltkNegLex]
porter = PorterStemmer()
stemmedPosLex = set(lexStemmer(posLex))
stemmedNegLex = set(lexStemmer(negLex))
print(stemmedPosLex)
print(stemmedNegLex)
print(porter.stem("working"))

