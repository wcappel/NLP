import torch
import random
from pathlib import Path
from transformers import BertTokenizer, BertModel, BertForNextSentencePrediction


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


# Filters tokens to not get duds
def filterTokens(tokens):
    newTokens = []
    for lyric in tokens:
        if len(lyric) >= 4:
            newTokens.append(lyric)


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

goodInput = False
inputGenre = ""
while not goodInput:
    inputGenre = input("Enter the genre of lyrics you wish to generate (pop, country, metal, or rock): ")
    if type(inputGenre) is str:
        inputGenre = inputGenre.lower()
        if (inputGenre == 'pop') or (inputGenre == 'country') or (inputGenre == 'metal') or (inputGenre == 'rock'):
            break
        else:
            print("Only options are: pop, rock, metal, or country.")
    else:
        print("Only options are: pop, rock, metal, or country.")

selectedLyrics = []
if inputGenre == 'pop':
    selectedLyrics = popTokenized
elif input == 'rock':
    selectedLyrics = rockTokenized
elif input == 'metal':
    selectedLyrics = metalTokenized
else:
    selectedLyrics = countryTokenized

filterTokens(selectedLyrics)
randomNum = random.randint(1, len(selectedLyrics))
newRandomNum = random.randint(1, len(selectedLyrics))
while (newRandomNum == randomNum):
    newRandomNum = random.randint(1, len(selectedLyrics))
initialLyric = selectedLyrics[randomNum]
otherLyric = selectedLyrics[newRandomNum]
print(initialLyric)

initialIndexedTokens = tokenizer.convert_tokens_to_ids(initialLyric)
initialSegmentsIDs = [1] * len(initialLyric)
initialTokensTensor = torch.tensor([initialIndexedTokens])
initialSegmentsTensor = torch.tensor([initialSegmentsIDs])

model = BertForNextSentencePrediction('bert-base-uncased', output_hidden_states=True)
model.eval()





