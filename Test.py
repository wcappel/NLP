import torch
import random
from pathlib import Path
from transformers import BertTokenizer, BertModel, BertForNextSentencePrediction
from transformers import Trainer, TrainingArguments

# bert_model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased',output_hidden_states=True)

