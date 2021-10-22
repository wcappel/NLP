import torch
import random
from pathlib import Path
from transformers import BertTokenizer, BertModel, BertForNextSentencePrediction
from transformers import Trainer, TrainingArguments, TextDatasetForNextSentencePrediction, DataCollatorForLanguageModeling



def formatTuningFile(filePath):
    with open("tuning.txt", mode="w") as output:
        for file in filePath:
            current = open(file, 'r')
            text = current.read()
            text = text.split("\n")
            addNL = False
            if len(text) % 2 != 0:
                addNL = True
            current.close()
            for i, line in enumerate(text):
                if addNL and line is text[-1]:
                    continue
                elif i % 2 != 0:
                    output.write(line)
                    output.write("\n\n")
                else:
                    output.write(line)
                    output.write("\n")


# Tokenizes each lyric along w/ special differentiating tokens
def tokenizeLyrics(lyrics, tokenizer):
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

# Takes user input to choose genre of lyrics
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
    selectedLyrics = popFiles
elif input == 'rock':
    selectedLyrics = rockFiles
elif input == 'metal':
    selectedLyrics = metalFiles
else:
    selectedLyrics = countryFiles

# Randomly shuffles songs for tuning and generating
random.shuffle(selectedLyrics)
generatorFiles = selectedLyrics[0:(int)(len(selectedLyrics)/5)]
tuningFiles = selectedLyrics[(int)(len(selectedLyrics)/5):]

# Formats lyrics to put in text file to tune BERT
formatTuningFile(countryFiles)

# trainingArguments = TrainingArguments(output_dir="./tuningOutput", overwrite_output_dir=True)
# bertModel = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# tuningDataset = TextDatasetForNextSentencePrediction(tokenizer=tokenizer, file_path="tuning.txt", block_size=64)
# dataCollator = DataCollatorForLanguageModeling(tokenizer=tokenizer)
# trainer = Trainer(model=bertModel, args=trainingArguments, train_dataset=tuningDataset, tokenizer=tokenizer, data_collator=dataCollator)
# print("Ignore that error message.")
# trainer.train()
# trainer.save_model("./tuningOutput")

model = BertForNextSentencePrediction.from_pretrained("./tuningOutput")
tokenizer = BertTokenizer.from_pretrained("./tuningOutput")
lyricEx = ["As he sang his song of hope for the last time"]
tokenizedLyricList = tokenizeLyrics(lyricEx, tokenizer)
tokenizedLyric = tokenizedLyricList[0]
print(tokenizedLyric)
initialIndexedTokens = tokenizer.convert_tokens_to_ids(tokenizedLyric)
initialSegmentsIDs = [1] * len(tokenizedLyric)
initialTokensTensor = torch.tensor([initialIndexedTokens])
initialSegmentsTensor = torch.tensor([initialSegmentsIDs])
outputs = model(initialSegmentsTensor, initialTokensTensor)
print(outputs)