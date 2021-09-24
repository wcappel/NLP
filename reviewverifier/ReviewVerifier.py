
import nltk
import ssl
import string
import random
import collections
import nltk.metrics
from nltk.metrics.scores import (precision, recall)
from pathlib import Path
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("punkt")
nltk.download("stopwords")

# Reads each file in directory and adds to list that will be returned
def readFiles(filePath):
    strList = []
    for file in filePath:
        current = open(file, 'r')
        text = current.read()
        current.close()
        strList.append(text)
    return strList

# 'Main' starts here:

# File paths for each labeled directory w/ only txt files selected
negFolder = Path('./neg/').rglob('*.txt')
posFolder = Path('./pos/').rglob('*.txt')

# Lists of every txt file in each folder
negFiles = [x for x in negFolder]
posFiles = [y for y in posFolder]

labeledPosReviews = readFiles(posFiles)
labeledNegReviews = readFiles(negFiles)
labeledReviews = []

#print(labeledPosReviews)

noPunctPos = []
noPunctNeg = []

# Loops case fold documents and remove punctuation
for document in labeledPosReviews:
    document = document.lower()
    document = "".join([char for char in document if char not in string.punctuation])
    labeledReviews.append((document, "pos"))

for document in labeledNegReviews:
    document = document.lower()
    document = "".join([char for char in document if char not in string.punctuation])
    labeledReviews.append((document, "neg"))

#print(labeledReviews)

# Tokenization
#words = word_tokenize(text)
tokens = set(word for words in labeledReviews for word in word_tokenize(words[0]))
#print(tokens)
stopwords = stopwords.words("english")

#Finish tokenization
#data = [({word: (word in word_tokenize(x[0])) for word in tokens}, x[1]) for x in labeledReviews]
data = []
porter = PorterStemmer()
for x in labeledReviews:
    dictionary = {}
    for word in tokens:
        if word not in stopwords:
            con = word in x[0]
            stemmed = porter.stem(word)
            if not dictionary.get(stemmed):
                dictionary[stemmed] = con
    data.append((dictionary, x[1]))




#print(data)
# Randomizing and splitting data for training and testing
random.shuffle(data)
training = data[0:(int)(len(labeledReviews)/2)]
testing = data[(int)(len(labeledReviews)/2):]

# NB Classifer
classifier = nltk.NaiveBayesClassifier.train(training)
classifier.show_most_informative_features()

truesets = collections.defaultdict(set)
classifiersets =  collections.defaultdict(set)
# you want to look at precision and recall in both training anf testing
# if your performnace is really good in training but horrible in testing
# that means your model is overfitted

for i, (doc, label) in enumerate(testing):
  #run your classifier over testing dataset to see the peromance
  truesets[label].add(i)
  observed = classifier.classify(doc)
  classifiersets[observed].add(i)

print(truesets)
print(classifiersets)
# [1, 3, 4, 5, 19, 45]
# [2, 3, 4, 19, 25, 40]

pos_precision = precision(truesets['pos'], classifiersets['pos'])
neg_precision = precision(truesets["neg"], classifiersets["neg"])
pos_recall = recall(truesets['pos'], classifiersets['pos'])
neg_recall = recall(truesets["neg"], classifiersets["neg"])
print("Positive precision: ")
print(pos_precision)
print("Negative precision: ")
print(neg_precision)
print("Positive recall: ")
print(pos_recall)
print("Negative recall: ")
print(neg_recall)
