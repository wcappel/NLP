import nltk
import re
from nltk.tokenize import word_tokenize
from pathlib import Path
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("punkt")

# Reads each file in directory and adds to list that will be returned
def readFiles(filePath):
    strList = []
    for file in filePath:
        current = open(file, 'r')
        text = current.read()
        current.close()
        strList.append(text)
    return strList
# 'main'

# File paths for each labeled directory w/ only txt files selected
negFolder = Path('./neg/').rglob('*.txt')
posFolder = Path('./pos/').rglob('*.txt')

# Lists of every txt file in each folder
negFiles = [x for x in negFolder]
posFiles = [y for y in posFolder]

labeledPosReviews = readFiles(posFiles)
labeledNegReviews = readFiles(negFiles)
labeledReviews = {}

for document in labeledPosReviews:
    re.sub(r'\\n', '', document)
    labeledReviews.update({document.lower(): "pos"})

for document in labeledNegReviews:
    document = re.sub(r'\\n', '', document)
    labeledReviews.update({document.lower(): "neg"})

#print(labeledReviews)

# Tokenization
tokens = set(word for words in labeledReviews for word in word_tokenize(words[0]))
print(tokens)





