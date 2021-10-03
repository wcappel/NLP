import nltk
import string
import random
import collections
import nltk.metrics
import pandas
from nltk.metrics.scores import (precision, recall)
from pathlib import Path
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.corpus import opinion_lexicon
from sklearn.linear_model import LogisticRegression
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


# Function case folds documents and remove punctuation
def removePunct(reviews):
    result = []
    for document in reviews:
        document = document.lower()
        document = "".join([char for char in document if char not in string.punctuation])
        result.append(document)
    return result


# Formats data for NLTK NB classifier
def formatForNB(reviews):
    result = []
    for x in reviews:
        doc_tokens = word_tokenize(x[0])
        dictionary = {word: (word in doc_tokens) for word in trainTokens}
        newTup = (dictionary, x[1])
        result.append(newTup)
    return result


# Method that formats dataset for LR
def reformatForLR(notFormatted):
    reformatted = []
    for rev in notFormatted:
        bigList = [word_tokenize(rev[0]), rev[1]]
        reformatted.append(bigList)
    return reformatted


# Given a review, returns list of features counts with class appended
# Parameter 'review' will look like: (['A', 'sentence'], pos)
def featureCount(x):
    frequencies = [0, 0, 0, 0, 0, 0, 0]
    for w in x[0]:
        if w in stemmedPosLex:
            frequencies[0] += 1
        elif w in stemmedNegLex:
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
    if x[1] == 'pos':
        frequencies[6] = 1
    elif x[1] == 'neg':
        frequencies[6] = 0
    # print(frequencies)
    return frequencies


# 'Main' starts here:
# Lexicons for LR features:
print("formatting lexicons...")
porter = PorterStemmer()
nltkPosLex = opinion_lexicon.positive()
nltkNegLex = opinion_lexicon.negative()
posLex = ["".join(list_of_words) for list_of_words in nltkPosLex]
negLex = ["".join(list_of_words) for list_of_words in nltkNegLex]

stemmedPosLex = set(lexStemmer(posLex))
stemmedNegLex = set(lexStemmer(negLex))

# File paths for each labeled directory w/ only txt files selected
negFolder = Path('./neg/').rglob('*.txt')
posFolder = Path('./pos/').rglob('*.txt')
posRatingsPath = Path('./ratings/').rglob('positive.txt')
negRatingsPath = Path('./ratings/').rglob('negative.txt')


# Lists of every txt file in each folder
print("reading files...")
negFiles = [x for x in negFolder]
posFiles = [y for y in posFolder]
posRatingsFile = [pos for pos in posRatingsPath]
negRatingsFile = [neg for neg in negRatingsPath]

labeledPosReviews = readFiles(posFiles)
labeledNegReviews = readFiles(negFiles)
posRatings = readFiles(posRatingsFile)
negRatings = readFiles(negRatingsFile)
print(posRatings)

# Split ratings from each file into list w/ # and rating
print("preprocessing data...")
posRatingsSplit = posRatings[0].split("\n")
negRatingsSplit = negRatings[0].split("\n")
posSepRatings = []
negSepRatings = []
for rating in posRatingsSplit:
    posSepRatings.append(str.split(" "))

for rating in negRatingsSplit:
    negSepRatings.append(str.split(" "))

ratings = posSepRatings + negSepRatings

# Case fold and remove punct.
posReviews = removePunct(labeledPosReviews)
negReviews = removePunct(labeledNegReviews)

# Remove stop words
labeledReviews = []
stopWords = stopwords.words()
for document in posReviews:
    for word in document:
        if word not in stopWords:
            newDoc = " ".join(porter.stem(word))
    labeledReviews.append((document, "pos"))

for document in negReviews:
    for word in document:
        if word not in stopWords:
            newDoc = " ".join(porter.stem(word))
    labeledReviews.append((document, "neg"))
# print(labeledReviews)

# Randomizing and splitting data for training and testing, use same random value for ratings
seed = random.randint(0, 2147483647)
random.seed(seed)
random.shuffle(labeledReviews)
random.seed(seed)
random.shuffle(ratings)
training = labeledReviews[0:(int)(len(labeledReviews)/2)]
testing = labeledReviews[(int)(len(labeledReviews)/2):]
ratingsTesting = ratings[(int)(len(ratings)/2):]

# Generating tokens
trainTokens = set(word for words in training for word in word_tokenize(words[0]))
testTokens = set(word for words in testing for word in word_tokenize(words[0]))

# Tokenizing and formatting for NB
print("started formatting data for NB classifier...")
nbTrainData = formatForNB(training)
nbTestData = formatForNB(testing)
print("finished formatting data for NB.")

# NB classifer training w/ training dataset
print("training NB classifier...")
classifier = nltk.NaiveBayesClassifier.train(nbTrainData)

# Most informative features from NB classifier
classifier.show_most_informative_features()

# Run NB classifer over testing dataset
print("testing NB classifier...")
truesets = collections.defaultdict(set)
classifiersets = collections.defaultdict(set)
for i, (doc, label) in enumerate(nbTestData):
  truesets[label].add(i)
  observed = classifier.classify(doc)
  classifiersets[observed].add(i)

# Shows true and classifer sets
#print(truesets)
#print(classifiersets)

# Calculate positive/negative precision and recall
print("evaluating classifier...")
print(" ################################## NB Results: ################################## ")
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

# Feature table for LR
'''
Feature:    Definition:                 Initial weight:
f1          word ∈ pos. lexicon         1
f2          word ∈ neg. lexicon         -1
f3          bigrams "not good"          -3
f4          bigrams "i like"            2
f5          bigrams "not bad"           3
f6          bigrams "dont like"         -3
'''

# Using the same split, reformat training and testing data for LR classifer
print("formatting data for LR...")
lrTraining = reformatForLR(training)
lrTesting = reformatForLR(testing)
print("finished formatting data for LR.")

print("counting features for LR classifier...")
formattedTraining = []
for review in lrTraining:
    formattedTraining.append(featureCount(review))

formattedTesting = []
for review in lrTesting:
    formattedTesting.append(featureCount(review))

# Turning data into DataFrames w/ labeled feature columns
print("building dataframes...")
trainFrame = pandas.DataFrame(formattedTraining, columns=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'class'])
testFrame = pandas.DataFrame(formattedTesting, columns=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'class'])
#print(trainFrame)

# Get doc. # w/ feature columns from dataframe
xTrain = trainFrame.iloc[:, 0:6]
xTest = testFrame.iloc[:, 0:6]

# Get doc. # w/ class tag from dataframe
yTrain = trainFrame.iloc[:, -1]
yTest = testFrame.iloc[:, -1]
uncutTestTruePos = testFrame.loc[testFrame['class'] == 1]
uncutTestTrueNeg = testFrame.loc[testFrame['class'] == 0]
testTruePos = uncutTestTruePos.iloc[:, 0:6]
testTrueNeg = uncutTestTrueNeg.iloc[:, 0:6]
#print(testTruePos)

# print("Doc. # w/ feature counts:")
# print(xTrain)
# print("Doc. # w/ class:")
# print(yTrain)

# Train LR classifier on data
print("beginning logistic regression...")
logRegression = LogisticRegression(solver='sag') # class_weight=[1, -1, -3, 2, 3, -3]
logRegression.fit(xTrain, yTrain)
print("logistic regression training done.")

# Testing LR classifier
print("testing LR classifier...")
predictedFromTruePos = logRegression.predict(testTruePos)
predictedFromTrueNeg = logRegression.predict(testTrueNeg)
print("finished testing LR classifier.")

# Comparing LR predicted classes w/ true classes
truePositives = 0
falsePositives = 0
trueNegatives = 0
falseNegatives = 0

for result in predictedFromTruePos:
    if result == 0:
        falseNegatives += 1
    else:
        truePositives += 1

for result in predictedFromTrueNeg:
    if result == 0:
        trueNegatives += 1
    else:
        falsePositives += 1

# Count true/false positives and negatives; calculate precision, recall, and f-measure
print(" ################################## LR Results: ################################## ")
print("# of true positives: " + str(truePositives) + ", # of false positives: " +
      str(falsePositives) + ", # of true negatives: " + str(trueNegatives) +
      ", # of false negatives: " + str(falseNegatives))

lrPosPrec = truePositives / (truePositives + falsePositives)
lrNegPrec = trueNegatives / (trueNegatives + falseNegatives)
lrPosRecall = truePositives / (truePositives + falseNegatives)
lrNegRecall = trueNegatives / (trueNegatives + falsePositives)
lrPosF = (2 *(lrPosPrec * lrPosRecall)) / (lrPosPrec + lrPosRecall)
lrNegF = (2 *(lrNegPrec * lrNegRecall)) / (lrNegPrec + lrNegRecall)
print("Positive Precision: " + str(lrPosPrec))
print("Negative Precision: " + str(lrNegPrec))
print("Positive Recall: " + str(lrPosRecall))
print("Negative Recall: " + str(lrNegRecall))
print("Positive F-Measure: " + str(lrPosF))
print("Negative F-Measure: " + str(lrNegF))
print("         Ignore warning below         ")

