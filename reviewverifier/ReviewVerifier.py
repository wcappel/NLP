import nltk
import ssl
import string
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

# Loops case fold documents and remove punctuation, then append to labeledReviews based on tagged class
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
tokenized = set(word for words in labeledReviews for word in word_tokenize(words[0]))
#print(tokenized)

#Remove stop words
stopwords = stopwords.words("english")
removedSW = [word for word in tokenized if word not in stopwords]
#print(removedSW)

#Stemming
porter = PorterStemmer()
stemmedWords = [porter.stem(word) for word in removedSW]
print(stemmedWords)






