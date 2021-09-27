import nltk
import string
import random
import collections
import nltk.metrics
import math
import numpy
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
for document in labeledReviews:
    dictionary = {}
    for word in tokens:
        if word not in stopwords:
            valid = word in document[0]
            stemmed = porter.stem(word)
            if not dictionary.get(stemmed):
                dictionary[stemmed] = valid
    data.append((dictionary, document[1]))

#print(data)

# Randomizing and splitting data for training and testing
random.shuffle(data)
training = data[0:(int)(len(labeledReviews)/2)]
testing = data[(int)(len(labeledReviews)/2):]

# Example of data format to work with
#testData = [({'This': False, 'is': False,'a': False, 'sentence':False}, 'pos'), ({'Another': False, 'sentence': False}, 'pos')]


# Using the same split, reformat training and testing data for LR classifer
lrTraining = reformatForLR(training)
lrTesting = reformatForLR(testing)

#print(lrTraining)
#print(len(lrTraining))

# NB classifer training w/ training dataset
classifier = nltk.NaiveBayesClassifier.train(training)

# Most informative features from NB classifier
classifier.show_most_informative_features()

truesets = collections.defaultdict(set)
classifiersets =  collections.defaultdict(set)

# Run NB classifer over testing dataset
for i, (doc, label) in enumerate(testing):
  #run your classifier over testing dataset to see the peromance
  truesets[label].add(i)
  observed = classifier.classify(doc)
  classifiersets[observed].add(i)

# Shows true and classifer sets
print(truesets)
print(classifiersets)

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


# z = w * x + b
def calculateZ(w, x, b):
    dProd = numpy.dot(w, x)
    return dProd + b


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


# classVal is either 1 (pos) or 0 (neg),
def probForClass(classVal, z):
    if classVal == 1:
        return sigmoid(z)
    elif classVal == 0:
        return 1.0 - sigmoid(z)
    else:
        print("classVal must be 1 or 0")


# Estimates class based on greater prob. of either pos or neg
def estimateClass(z):
    if probForClass(1, z) > probForClass(0, z):
        return 1
    else:
        return 0


# Cost function:
def calculateCost(estimatedProb, trueClass):
    return -((trueClass * math.log(estimatedProb)) + ((1 - trueClass) * math.log(1 - estimatedProb)))


#print(calculateCost(0.69, 0))

# theta = (old) theta + stepSize * gradient
# Gradient descent starts here:

# xj = array of freq. of features, yActual = actual class (0 or 1)
#def gradientDesc(z, xj, trueClass):


#stepSizes = [0.01, 0.05, 0.1, 0.5, 1] ?

