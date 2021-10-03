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
list1 = ['haha', 'yes', 'no', 'maybe', 'so']
list2 = ['haha', 'yes', 'no', 'maybe', 'so']
seed = random.randint(0, 2147483647)
random.seed(seed)
random.shuffle(list1)
random.seed(seed)
random.shuffle(list2)
print(list1)
print(list2)