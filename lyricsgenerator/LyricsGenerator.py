import torch
import random
from pathlib import Path
from transformers import BertTokenizer, BertForPreTraining, Trainer, TrainingArguments
from transformers import TextDatasetForNextSentencePrediction, DataCollatorForLanguageModeling


# Formats lyrics from genre directory into format to train model
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


# Reads each file in directory and adds to list that will be returned along w/ file #
def readFiles(filePath):
    files = []
    for file in filePath:
        current = open(file, 'r')
        text = current.read()
        current.close()
        files += text.split("\n")
    return files


# Returns the 30 most likely predictions for the masked word in the lyric
def predictMasked(maskedLyric):
    encodings = tokenizer.encode(maskedLyric, return_tensors='pt')
    maskedPos = (encodings.squeeze() == tokenizer.mask_token_id).nonzero().item()
    with torch.no_grad():
        output = model(encodings)
    hiddenState = output[0].squeeze()
    maskHiddenState = hiddenState[maskedPos]
    ids = torch.topk(maskHiddenState, k=30, dim=0)[1]
    predictedWords = []
    for id in ids:
        word = tokenizer.convert_ids_to_tokens(id.item())
        predictedWords.append(word)
    return predictedWords


# Generates a one line lyric from a sample lyric
def genSingleLyric(initialLyric):
    genLength = random.randint(2, 7)
    count = genLength
    nextLyric = ""
    prevLyric = []
    while count > 0:
        maskedLyric = initialLyric + " [MASK]"
        predicted = predictMasked(maskedLyric)
        random.shuffle(predicted)
        for word in predicted:
            if len(word) > 1 and word not in prevLyric:
                prevLyric.append(word)
                if "##" in word:
                    if count != genLength:
                        nextLyric += word[2:]
                        initialLyric += word[2:]
                    else:
                        count += 1
                else:
                    nextLyric += " " + word
                    initialLyric += " " + word
                break
        count -= 1
    return nextLyric


# Generates multiple lyrics sequentially
def genMultipleLyrics(numLyrics):
    goodSelection = False
    while goodSelection is False:
        sampleLyric = sampleLyrics[random.randint(1, len(sampleLyrics))]
        if len(sampleLyric.split()) > 3:
            goodSelection = True
    lyrics = ""
    while numLyrics > 0:
        generatedLyric = genSingleLyric(sampleLyric)
        generatedLyric = generatedLyric[1].upper() + generatedLyric[2:]
        lyrics += generatedLyric + "\n"
        sampleLyric = generatedLyric
        numLyrics -= 1
    return lyrics


# Outputs score from model of how likely the generated sentence is to follow the initial sentence
def getLyricScore(initialLyric, genLyric):
    encoding = tokenizer(initialLyric, genLyric, return_tensors='pt')
    outputs = model(**encoding, labels=torch.LongTensor([1]))
    nspLogits = outputs.seq_relationship_logits  # use seq_relationship_logits for NSP, use prediction_logits for MLM
    #print(nspLogits[0, 0] < nspLogits[0, 1]) # [0, 0] is the predicted sentence, False means it is natural
    return nspLogits[0, 0].item()


# Then this too?
def getBestLyric(initialLyric, possibleLyrics):
    lyricScores = []
    for lyric in possibleLyrics:
        lyricScores.append(getLyricScore(initialLyric, lyric))
    maxScore = lyricScores[0]
    maxScoreIndex = 0
    for j, score in enumerate(lyricScores):
        if score > maxScore:
            maxScoreIndex = j
    return possibleLyrics[maxScoreIndex]


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
    trainingArguments = TrainingArguments(output_dir="./popModel", overwrite_output_dir=True)
elif inputGenre == 'rock':
    selectedLyrics = rockFiles
    trainingArguments = TrainingArguments(output_dir="./rockModel", overwrite_output_dir=True)
elif inputGenre == 'metal':
    selectedLyrics = metalFiles
    trainingArguments = TrainingArguments(output_dir="./metalModel", overwrite_output_dir=True)
else:
    selectedLyrics = countryFiles
    trainingArguments = TrainingArguments(output_dir="./countryModel", overwrite_output_dir=True)

# Splits songs for tuning and generating
generatorFiles = selectedLyrics[0:(int)(len(selectedLyrics)/5)]
tuningFiles = selectedLyrics[(int)(len(selectedLyrics)/5):]

# Formats lyrics to put in text file to tune BERT
formatTuningFile(selectedLyrics)

# Read files for samples to generates lyrics
sampleLyrics = readFiles(generatorFiles)

# Training model on data with masking
'''Remember to stick training code in another file or tell her it's commented out'''
# bertModel = BertForPreTraining.from_pretrained('bert-base-uncased')
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# tuningDataset = TextDatasetForNextSentencePrediction(tokenizer=tokenizer, file_path="tuning.txt", block_size=256)
# dataCollator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
# trainer = Trainer(model=bertModel, args=trainingArguments, train_dataset=tuningDataset, tokenizer=tokenizer, data_collator=dataCollator)
# trainer.train()
# if inputGenre == 'pop':
#     trainer.save_model("./popModel")
# elif inputGenre == "rock":
#     trainer.save_model("./rockModel")
# elif inputGenre == "metal":
#     trainer.save_model("./metalModel")
# else:
#     trainer.save_model("./countryModel")

# Selecting model that was saved (have to use new string literal every time bc. the parameter is dumb lol)
if inputGenre == 'pop':
    model = BertForPreTraining.from_pretrained("./popModel")
    tokenizer = BertTokenizer.from_pretrained("./popModel")
elif inputGenre == "rock":
    model = BertForPreTraining.from_pretrained("./rockModel")
    tokenizer = BertTokenizer.from_pretrained("./rockModel")
elif inputGenre == "metal":
    model = BertForPreTraining.from_pretrained("./metalModel")
    tokenizer = BertTokenizer.from_pretrained("./metalModel")
else:
    model = BertForPreTraining.from_pretrained("./countryModel")
    tokenizer = BertTokenizer.from_pretrained("./countryModel")


model.eval()


print(genMultipleLyrics(4))