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
    noPunctPos.append(document)

for document in labeledNegReviews:
    document = document.lower()
    document = "".join([char for char in document if char not in string.punctuation])
    noPunctNeg.append(document)

#print(noPunctNeg)

# Tokenization
#words = word_tokenize(text)
tokenizedNeg = []
tokenizedPos = []
for document in noPunctPos:
    tokens = word_tokenize(document)
    tokenizedPos.append(tokens)

for document in noPunctNeg:
    tokens = word_tokenize(document)
    tokenizedNeg.append(tokens)

#print(tokenizedPos)

# Remove stop words
stopwords = stopwords.words("english")
removedPos = []
removedNeg = []
for document in tokenizedPos:
    newDoc = []
    for word in document:
        if word not in stopwords:
            newDoc.append(word)
    removedPos.append(newDoc)

for document in tokenizedNeg:
    newDoc = []
    for word in document:
        if word not in stopwords:
            newDoc.append(word)
    removedNeg.append(newDoc)

#print(removedPos)

# Stemming
porter = PorterStemmer()
stemmedPos = []
stemmedNeg = []

for document in removedPos:
    newDoc = []
    for word in document:
        newDoc.append(porter.stem(word))
    stemmedPos.append(newDoc)

for document in removedNeg:
    newDoc = []
    for word in document:
        newDoc.append(porter.stem(word))
    stemmedNeg.append(newDoc)

#print(stemmedPos)

# Collating to list with tag
for document in stemmedPos:
  labeledReviews.append((document, "pos"))
for document in stemmedNeg:
  labeledReviews.append((document, "neg"))

print(labeledReviews)

# Randomizing and splitting data for training and testing
random.shuffle(labeledReviews)
training = labeledReviews[0:(int)(len(labeledReviews)/2)]
testing = labeledReviews[(int)(len(labeledReviews)/2):]
