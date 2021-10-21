import torch
from pathlib import Path
from transformers import BertTokenizer, BertModel


# Reads each text line of file in directory and adds to a list passed as a parameter
def readFiles(filePath, files):
    for file in filePath:
        current = open(file, 'r')
        text = current.read()
        text = text.split("\n")
        current.close()
        for line in text:
            files.append(line)
    return files


# Tokenizes each lyric along w/ special differentiating tokens
def tokenizeLyrics(lyrics):
    tokenizedLyrics = []
    for lyric in lyrics:
        lyric = "[CLS] " + lyric + " [SEP]"
        tokenizedLyrics.append(tokenizer.tokenize(lyric))
    return tokenizedLyrics


# File paths for each directory of lyrics
countryFolder = Path('./Country/').rglob('*.txt')
metalFolder = Path('./Metal/').rglob('*.txt')
popFolder = Path('./Pop/').rglob('*.txt')
rockFolder = Path('./Rock/').rglob('*.txt')

# Lists of every txt file in each folder
countryFiles = [x for x in countryFolder]
metalFiles = [x for x in metalFolder]
popFiles = [x for x in popFolder]
rockFiles = [x for x in rockFolder]

# Adds all lyrics by line in file to list
countryLyrics = []
metalLyrics = []
popLyrics = []
rockLyrics = []
readFiles(countryFiles, countryLyrics)
readFiles(metalFiles, metalLyrics)
readFiles(popFiles, popLyrics)
readFiles(rockFiles, rockLyrics)

# Remove stop words and punctuation? Prob. not bc. will be in same place in embeddings
# Tokenize lyrics
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
countryTokenized = tokenizeLyrics(countryLyrics)
metalTokenized = tokenizeLyrics(metalLyrics)
popTokenized = tokenizeLyrics(popLyrics)
rockTokenized = tokenizeLyrics(rockLyrics)




