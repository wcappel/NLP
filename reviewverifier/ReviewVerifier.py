
import nltk
import ssl
import string
import random
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




print(data)
# Randomizing and splitting data for training and testing
random.shuffle(data)
training = data[0:(int)(len(labeledReviews)/2)]
testing = data[(int)(len(labeledReviews)/2):]

# NB Classifer
classifier = nltk.NaiveBayesClassifier.train(training)
classifier.show_most_informative_features()
