import torch
import random
import sentencepiece                                          # Please import this one as well
from pathlib import Path
from transformers import BertTokenizer, BertForPreTraining, Trainer, TrainingArguments
from transformers import TextDatasetForNextSentencePrediction, DataCollatorForLanguageModeling
from transformers import FNetForNextSentencePrediction, FNetTokenizer


# Formats lyrics from genre directory into format to train model
def formatTuningFile(filePath, outputName):
    with open(outputName, mode="w") as output:
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
    lyrics = []
    goodSelection = False
    while goodSelection is False:
        sampleLyric = sampleLyrics[random.randint(0, len(sampleLyrics) - 1)]
        if len(sampleLyric.split()) > 3:
            goodSelection = True
    while numLyrics > 0:
        generatedLyric = genSingleLyric(sampleLyric)
        generatedLyric = generatedLyric[1].upper() + generatedLyric[2:]
        lyrics.append(generatedLyric)
        sampleLyric = generatedLyric
        numLyrics -= 1
    return lyrics


# Outputs 'False' if evaluation model considers the generated lyric a naturally following lyric
def getEvalScore(initialLyric, evalLyric):
    encoding = evalTokenizer(initialLyric, evalLyric, return_tensors='pt')
    outputs = evalModel(**encoding, labels=torch.LongTensor([1]))
    evalLogits = outputs.logits
    return evalLogits[0, 0].item() < evalLogits[0, 1].item()


# Generates and then evaluates a batch of lyrics against their sample lyric
def evaluate(size):
    evalList = []
    while size > 0:
        goodSelection = False
        sampleLyric = ""
        while goodSelection is False:
            sampleLyric = sampleLyrics[random.randint(0, len(sampleLyrics) - 1)]
            if len(sampleLyric.split()) > 3:
                goodSelection = True
        toEval = genSingleLyric(sampleLyric)
        evalList.append(getEvalScore(sampleLyric, toEval))
        size -= 1
    return evalList




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

# Setting up training arguments and selecting files of request genre
selectedLyrics = []
if inputGenre == 'pop':
    selectedLyrics = popFiles
    trainingArguments = TrainingArguments(output_dir="./popModel", overwrite_output_dir=True)
    evalArguments = TrainingArguments(output_dir="./popEval", overwrite_output_dir=True)
elif inputGenre == 'rock':
    selectedLyrics = rockFiles
    trainingArguments = TrainingArguments(output_dir="./rockModel", overwrite_output_dir=True)
    evalArguments = TrainingArguments(output_dir="./rockEval", overwrite_output_dir=True)
elif inputGenre == 'metal':
    selectedLyrics = metalFiles
    trainingArguments = TrainingArguments(output_dir="./metalModel", overwrite_output_dir=True)
    evalArguments = TrainingArguments(output_dir="./metalEval", overwrite_output_dir=True)
else:
    selectedLyrics = countryFiles
    trainingArguments = TrainingArguments(output_dir="./countryModel", overwrite_output_dir=True)
    evalArguments = TrainingArguments(output_dir="./countryEval", overwrite_output_dir=True)

# Splits songs for tuning and generating
generatorFiles = selectedLyrics[0:int(len(selectedLyrics) * 0.2)]
tuningFiles = selectedLyrics[int(len(selectedLyrics) * 0.2): int(len(selectedLyrics) * 0.6)]
evalFiles = selectedLyrics[int(len(selectedLyrics) * 0.6):]


# Formats lyrics to put in text file to tune BERT and FNet
print("formatting tuning and evaluation input files...")
formatTuningFile(tuningFiles, "tuning.txt")
formatTuningFile(evalFiles, "eval.txt")

# Read files for samples to generates lyrics
print("reading files for sample lyrics...")
sampleLyrics = readFiles(generatorFiles)

# Training BERT model on data with masking
print("TUNING CODE HAS BEEN COMMENTED OUT, IT IS BELOW THIS STATEMENT")
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

# Train evaluation model on different split of lyrics
print("TUNING CODE FOR EVALUATION MODEL HAS BEEN COMMENTED OUT, IT IS BELOW THIS STATEMENT")
# evalModel = FNetForNextSentencePrediction.from_pretrained('google/fnet-base')
# evalTokenizer = FNetTokenizer.from_pretrained('google/fnet-base')
# evalDataset = TextDatasetForNextSentencePrediction(tokenizer=evalTokenizer, file_path="eval.txt", block_size=256)
# evalDataCollator = DataCollatorForLanguageModeling(tokenizer=evalTokenizer)
# evalTrainer = Trainer(model=evalModel, args=evalArguments, train_dataset=evalDataset, tokenizer=evalTokenizer, data_collator=evalDataCollator)
# evalTrainer.train()
# if inputGenre == 'pop':
#     evalTrainer.save_model("./popEval")
# elif inputGenre == "rock":
#     evalTrainer.save_model("./rockEval")
# elif inputGenre == "metal":
#     evalTrainer.save_model("./metalEval")
# else:
#     evalTrainer.save_model("./countryEval")

# Selecting evaluation model that was saved
print("selecting models that were saved from training...")
if inputGenre == 'pop':
    model = BertForPreTraining.from_pretrained("./popModel")
    tokenizer = BertTokenizer.from_pretrained("./popModel")
    evalModel = FNetForNextSentencePrediction.from_pretrained("./popEval")
    evalTokenizer = FNetTokenizer.from_pretrained("./popEval")
elif inputGenre == "rock":
    model = BertForPreTraining.from_pretrained("./rockModel")
    tokenizer = BertTokenizer.from_pretrained("./rockModel")
    evalModel = FNetForNextSentencePrediction.from_pretrained("./rockEval")
    evalTokenizer = FNetTokenizer.from_pretrained("./rockEval")
elif inputGenre == "metal":
    model = BertForPreTraining.from_pretrained("./metalModel")
    tokenizer = BertTokenizer.from_pretrained("./metalModel")
    evalModel = FNetForNextSentencePrediction.from_pretrained("./metalEval")
    evalTokenizer = FNetTokenizer.from_pretrained("./metalEval")
else:
    model = BertForPreTraining.from_pretrained("./countryModel")
    tokenizer = BertTokenizer.from_pretrained("./countryModel")
    evalModel = FNetForNextSentencePrediction.from_pretrained("./countryEval")
    evalTokenizer = FNetTokenizer.from_pretrained("./countryEval")

# Generate lyrics
print("generating a sample of lyrics...")
model.eval()
print("—————————————————————————————————————————————————————")
for lyric in genMultipleLyrics(4):
    print(lyric)
print("—————————————————————————————————————————————————————")

# Evaluate lyrics
print("evaluating lyrics...")
evalModel.eval()
evaluatedScores = evaluate(100)
passed = 0
failed = 0
for score in evaluatedScores:
    if score == False:
        passed += 1
    else:
        failed += 1

print("################################ EVALUATION ################################")
print("Evaluated " + str(len(evaluatedScores)) + " lines of generated lyrics. Amount of lyrics that "
        "passed: " + str(passed) + ". Amount of lyrics that failed: " + str(failed) + ".")
