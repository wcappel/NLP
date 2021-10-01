import nltk
import string
import random
import collections
import nltk.metrics
import math
import numpy
import pandas
from nltk.metrics.scores import (precision, recall)
from pathlib import Path
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.corpus import opinion_lexicon

#nltk.download("punkt")
#nltk.download("stopwords")
#nltk.download('opinion_lexicon')


# Reads each file in directory and adds to list that will be returned
def readFiles(filePath):
    strList = []
    for file in filePath:
        current = open(file, 'r')
        text = current.read()
        current.close()
        strList.append(text)
    return strList


# Method for stemming a list derived from nltk lexicons
def lexStemmer(lexicon):
    stemmedLexicon = []
    for word in lexicon:
        stemmedLexicon.append(porter.stem(word))
    return stemmedLexicon


# testData = [({'This': False, 'not': False,'good': False, 'sentence':False}, 'pos'), ({'Another': False, 'sentence': False}, 'neg')]
# Method that reformats nltk-NB-formatted dataset for our LR
def reformatForLR(nbFormatted):
    reformatted = []
    for x in nbFormatted:
        newList = []
        for key in x[0]:
            newList.append(key)
        bigList = [newList, x[1]]
        reformatted.append(bigList)
    return reformatted


# Given a review, returns list of features counts with class appended
# Parameter 'review' will look like: (['A', 'sentence'], pos)
def featureCount(review):
    frequencies = [0, 0, 0, 0, 0, 0, 0]
    for word in review[0]:
        if word in stemmedPosLex:
            frequencies[0] += 1
        elif word in stemmedNegLex:
            frequencies[1] += 1
    restrung = " ".join(review[0])
    reviewBigrams = list(nltk.bigrams(restrung.split()))
    for bigram in reviewBigrams:
        #print(bigram)
        if bigram[0] == 'not' and bigram[1] == 'good':
            frequencies[2] += 1
        elif bigram[0] == 'i' and bigram[1] == 'like':
            frequencies[3] += 1
        elif bigram[0] == 'not' and bigram[1] == 'bad':
            frequencies[4] += 1
        elif bigram[0] == 'dont' and bigram[1] == 'like':
            frequencies[5] += 1
    if review[1] == 'pos':
        frequencies[6] = 1
    elif review[1] == 'neg':
        frequencies[6] = 0
    # print(frequencies)
    return frequencies


# ! Potentially think about using ratings as a feature

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

# Debug

#print(labeledReviews)

# Tokenization
#words = word_tokenize(text)
tokens = set(word for words in labeledReviews for word in word_tokenize(words[0]))
#print(tokens)
stopwords = stopwords.words("english")

# Finish tokenization
porter = PorterStemmer()
#data = [({porter.stem(word): (word in word_tokenize(x[0])) for word in tokens if word not in stopwords}, x[1]) for x in labeledReviews]
#test   [({'This': False, 'is': False,'a': False, 'sentence':False}, 'pos'), ({'Another': False, 'sentence': False}, 'pos')]

data = []
for document in labeledReviews:
    dictionary = {}
    for word in tokens:
        if word not in stopwords:
            valid = word in document[0]
            stemmed = porter.stem(word)
            if not dictionary.get(stemmed):
                dictionary[stemmed] = valid
    data.append((dictionary, document[1]))
print("finished formatting data for NB")

# data = []
# for document in labeledReviews:
#     dictionary = {}
#     tokens = set(word for words in document)
#     for word in tokens:
#         if word not in stopwords:
#             if word in tokens:
#                 stemmed = porter.stem(word)
#                 dictionary[stemmed] = word in word_tokenize(document[0])
#     data.append((dictionary, document[1]))
# print("finished formatting data for NB")

#print(data)

# Randomizing and splitting data for training and testing
random.shuffle(data)
training = data[0:(int)(len(labeledReviews)/2)]
testing = data[(int)(len(labeledReviews)/2):]
#debugging = data[0:2]
#print(debugging)
#
# # Example of data format to work with
# #testData = [({'This': False, 'is': False,'a': False, 'sentence':False}, 'pos'), ({'Another': False, 'sentence': False}, 'pos')]
#
#
# # Using the same split, reformat training and testing data for LR classifer
lrTraining = reformatForLR(training)
lrTesting = reformatForLR(testing)
#lrDebugging = reformatForLR(debugging)
#print(lrTraining)
print(lrTraining)
print("finished formatting data for LR")

#print(lrTraining)
#print(len(lrTraining))

# # NB classifer training w/ training dataset
classifier = nltk.NaiveBayesClassifier.train(training)
#
# # Most informative features from NB classifier
# classifier.show_most_informative_features()
#
truesets = collections.defaultdict(set)
classifiersets =  collections.defaultdict(set)
#
# # Run NB classifer over testing dataset
for i, (doc, label) in enumerate(testing):
  #run your classifier over testing dataset to see the peromance
  truesets[label].add(i)
  observed = classifier.classify(doc)
  classifiersets[observed].add(i)
#
# # Shows true and classifer sets
# print(truesets)
# print(classifiersets)
#
# Calculate positive/negative precision and recall
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

# Lexicons for LR features:

nltkPosLex = opinion_lexicon.positive()
nltkNegLex = opinion_lexicon.negative()
posLex = ["".join(list_of_words) for list_of_words in nltkPosLex]
negLex = ["".join(list_of_words) for list_of_words in nltkNegLex]

stemmedPosLex = lexStemmer(posLex)
stemmedNegLex = lexStemmer(negLex)

# Classify fake reviews later

# Optimize for 85% precision, 78% recall ?
# Start w/ b == 0


#print(calculateCost(0.69, 0))

#Feature table
'''
Feature:    Definition:                 Initial weight:
f0          word ∈ pos. lexicon         1
f1          word ∈ neg. lexicon         -1
f2          bigrams "not good"          -3
f3          bigrams "i like"            2
f4          bigrams "not bad"           3
f5          bigrams "dont like"         -3
'''

initialWeights = [1, -1, -3, 2, 3, -3]

formattedTraining = []
for review in lrTraining:
    formattedTraining.append(featureCount(review))
# print(formattedTraining)

# formattedTesting = []
# for review in lrTesting:
#     formattedTesting.append(featureCount(re
#     view))

dataFrame = pandas.DataFrame(formattedTraining, columns=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'class'])
print(dataFrame)
print(len(lrTraining))


# theta = (old) theta + stepSize * gradient
# Gradient descent starts here:


'''
Probably want to perform gradient descent on each document with its cost function return value?
We then adjust weights using stepSizes/learning rate based on information such as feature
count in the document? But then probably also want to some kind of total value/accuracy across
all documents and use that to adjust.
So input for each doc. would probably be:
 weights = [w1, w2, ... wn]
 bias
 fCount = [f1, f2, ... fn]
 calculateCost return value
 estimated class
 true/actual class
 ... what else?
'''

# # z = w * x + b
# def calculateZ(w, x, b):
#     dProd = numpy.dot(w, x)
#     return dProd + b
#
# # Sigmoid funct. to fit btwn. 0 and 1
# def sigmoid(z):
#     return 1.0 / (1.0 + math.exp(-z))
#
#
# # classVal is either 1 (pos) or 0 (neg),
# def probForClass(classVal, z):
#     if classVal == 1:
#         return sigmoid(z)
#     elif classVal == 0:
#         return 1.0 - sigmoid(z)
#     else:
#         print("classVal must be 1 or 0")
#
#
# # Estimates class w/ prob. of neg and pos, using decision boundary
# def estimateClass(z):
#     if probForClass(1, z) > 0.5:
#         return 1
#     else:
#         return 0
#
#
# # Cost function:
# def calculateCost(trueClass, z):
#     return -((trueClass * math.log(sigmoid(z))) + ((1 - trueClass) * math.log(1 - sigmoid(z))))


# xj = array of freq. of features, trueClass = actual class (0 or 1)
#def gradientDesc(review, trueClass):


# stepSizes = [0.01, 0.05, 0.1, 0.5, 1] ?
# Use learning rate instead?
# learningRate = 0.1



