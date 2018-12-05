import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import config as _cfg
import numpy as np
from utils import *

class DataLoader:
	def __init__(self, source):
		self.load(source)

	def load(self, source):
		print("Reading data from {}".format(source))

		# max_len = 10 # trying to decrese sentence lengths...

		## load data
		data = open(source, encoding='utf-8').read()
		## split by /n/n
		dialogues = re.split(r'[\n\r]{2,}', data)
		## create context and response arrays
		context = []
		response = []
		for d in dialogues:
		    lines = d.split("\n")
		    for i in range(len(lines)):
		        # if i < len(lines)-1:
		        # if (i < len(lines)-1 
		        # 	and len(lines[i].split()) < max_len
		        # 	and len(lines[i+1].split()) < max_len
		        # 	):
		        if i < len(lines)-1:
		            context.append(lines[i])
		            response.append(lines[i+1])

		print("Loaded {} context-response pairs".format(len(context)))
		print()

		context = ["{} {} {}".format(_cfg.beg_token, line, _cfg.end_token) for line in context]
		response = ["{} {} {}".format(_cfg.beg_token, line, _cfg.end_token) for line in response]

		## If sequence length is not predefinde - set it here:
		#==========================================
		if _cfg.max_sequence_length == None:
			print("Sampling a max_sequence_length to account for 95% of the data...")
				
			l = [len(s.split()) for s in context]
			max_sequence_length = max(l)
			while max_sequence_length > 0:
				percent = 100*sum(i < max_sequence_length for i in l)/(len(context))
				if percent < 95:
					break
				max_sequence_length -= 1
			_cfg.max_sequence_length = max_sequence_length
			print("max_sequence_length set to {}".format(max_sequence_length))
		
		# ==========================================
		# Split into train and validation sets
		# ==========================================

		## split into train and validation sets
		num_validation = int(_cfg.part_val * len(context))
		print("Using {} samples for validation".format(num_validation))
		print("Shuffling data")
		idx = np.random.permutation(len(context))

		## generate encoder/decoder inputs
		print("Splitting into training and validation sets")
		encoder_input_train = []
		encoder_input_val = []
		decoder_input_train = []
		decoder_input_val = []
		for i in range(len(context)):
		    if i < num_validation:
		        encoder_input_val.append(context[idx[i]])
		        decoder_input_val.append(response[idx[i]])
		    else:
		        encoder_input_train.append(context[idx[i]])
		        decoder_input_train.append(response[idx[i]])


		# ==========================================
		# Tokenize
		# ==========================================

		print("\nUsing {} most frequent words".format(_cfg.num_tokens))
		self.tokenizer = Tokenizer(num_words=_cfg.num_tokens)
		self.tokenizer.fit_on_texts(list(encoder_input_train))

		
		# Save tokenizer
		file = 'model/tokenizer.pickle'
		print("saving tokenizer to {}".format(file))
		with open(file, 'wb') as handle:
		    pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


		print("Tokenizing all input data")
		encoder_input_train = self.tokenizer.texts_to_sequences(encoder_input_train)
		decoder_input_train = self.tokenizer.texts_to_sequences(decoder_input_train)

		encoder_input_val = self.tokenizer.texts_to_sequences(encoder_input_val)
		decoder_input_val = self.tokenizer.texts_to_sequences(decoder_input_val)


		# ==========================================
		# Pad Sequences
		# pad context sequences on left, pad response sequences on right
		# ==========================================
		
		pad_token = 0 #tokenizer.word_index["eos"]
		eos_token = 2
		print("Padding sequences to len = {}".format(_cfg.max_sequence_length))
		encoder_input_train = pad_sequences(encoder_input_train, maxlen=_cfg.max_sequence_length, padding='pre', value=pad_token)
		decoder_input_train = pad_sequences(decoder_input_train, maxlen=_cfg.max_sequence_length, padding='post', value=pad_token)

		encoder_input_val = pad_sequences(encoder_input_val, maxlen=_cfg.max_sequence_length, padding='pre', value=pad_token)
		decoder_input_val = pad_sequences(decoder_input_val, maxlen=_cfg.max_sequence_length, padding='post', value=pad_token)


		## Create decoder_target_train + decoder_target_val
		# these are same as decoder_input[train/val], but offset by 1 timestamp

		# TODO: shape of decoder_target sohuld be (None, _cfg.max_sequence_length, 1)
		print("Generating target matrices")
		decoder_target_train = np.full((np.shape(decoder_input_train)[0], np.shape(decoder_input_train)[1], 1), pad_token)
		for i in range(len(decoder_target_train)):
			for j in range(1,_cfg.max_sequence_length):
				decoder_target_train[i][j-1][0] = decoder_input_train[i][j]
		    
		decoder_target_val = np.full((np.shape(decoder_input_val)[0], np.shape(decoder_input_train)[1], 1), pad_token)
		for i in range(len(decoder_target_val)):
			for j in range(1,_cfg.max_sequence_length):
				decoder_target_val[i][j-1][0] = decoder_target_val[i][j]


		self.training = ([encoder_input_train, decoder_input_train], decoder_target_train)
		self.val = ([encoder_input_val, decoder_input_val], decoder_target_val)
		
		self.train_inputs = {
			"encoder": encoder_input_train,
			"decoder": decoder_input_train,
			"target": decoder_target_train,
		}
		self.val_inputs = {
			"encoder": encoder_input_val,
			"decoder": decoder_input_val,
			"target": decoder_target_val,
		}

	def get_training(self, n=None):
		# return ([encoder, decoder], target)
		if n is None:
			return self.training
		else:
			return ([self.train_inputs["encoder"][0:n], self.train_inputs["decoder"][0:n]], self.train_inputs["target"][0:n])

	def get_val(self, n = None):
		# return ([encoder, decoder], target)
		if n is None:
			return self.val
		else:
			return ([self.val_inputs["encoder"][0:n], self.val_inputs["decoder"][0:n]], self.val_inputs["target"][0:n])



