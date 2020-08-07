import tokenizers
import os
MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
EPOCHS = 10
#use bert multilingual model
BERT_PATH = "../input/bert_base_uncased/"
MODEL_PATH = "model.bin"
# TRAINING_FILE = "../input/imdb.csv"

#lower_case is set to true becoz model is uncased do case does not matters for this model
TOKENIZER = tokenizers.BertWordPieceTokenizer(
	os.path.join(BERT_PATH, 'vocab.txt'), 
	lowercase = True
)
