import config as _cfg
from keras.models import Model, load_model
from keras.layers import Input, Embedding, LSTM, Dense, Flatten
from keras.callbacks import ModelCheckpoint, Callback, LambdaCallback
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
import h5py, pickle
import numpy as np
import pandas as pd
from utils import *

queries = [
	"hi how are you",
	"alright",
	"my name is david. what is my name?",
	"my name is john. what is my name?",
	"are you a leader or a follower?",
	"are you a follower or a leader?",
	"what is moral?",
	"what is immoral?",
	"what is morality?",
	"what is the definition of altruism?",
	"ok ... so what is the definition of morality?",
	"tell me the definition of morality , i am quite upset now!"
]

class Decoder:
	def __init__(self, model, tokenizer):
		self.model = model
		self.tokenizer = tokenizer
		self.reverse_word_idx = {n:w for w,n in tokenizer.word_index.items()}
		self.reverse_word_idx[0] = "<UNK>"
		self.setup()
	
	def setup(self):
		""" Setup encoder + decoder models """
		decoder_lstm = self.model.get_layer("decoder_lstm")
		encoder_lstm = self.model.get_layer("encoder_lstm")
		decoder_dense = self.model.get_layer("decoder_dense")
		decoder_embed = self.model.get_layer("emb_decoder")
		encoder_embed = self.model.get_layer("emb_encoder")

		encoder_states = encoder_lstm.output[1:3]
		self.encoder_model = Model(encoder_embed.input, encoder_states)

		decoder_state_input_h = Input(shape=(_cfg.latent_dim,))
		decoder_state_input_c = Input(shape=(_cfg.latent_dim,))
		decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

		decoder_outputs, state_h, state_c = decoder_lstm(decoder_embed.output, initial_state=decoder_states_inputs)
		decoder_states = [state_h, state_c]
		decoder_outputs = decoder_dense(decoder_outputs)
		self.decoder_model = Model(
		    [decoder_embed.input] + decoder_states_inputs,
		    [decoder_outputs] + decoder_states)

	def text_to_seq(self, text):
		""" Convert string into integer input sequence """
		text = clean_str(text)
		tok = self.tokenizer.texts_to_sequences(["{} {} {}".format(_cfg.beg_token, text, _cfg.end_token)])
		padded = pad_sequences(tok, maxlen=_cfg.max_sequence_length, padding='pre', value=0)
		return padded

	def decode_batch(self, inp, file):
		f = open(file, "w")
		for context in inp:
			response = self.decode(context)
			f.write("{}\n{}\n".format(context, response))
		f.close()

	def decode(self, text):
		"""
		taken from 
		https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
		"""
		
		sequence = self.text_to_seq(text)
		# Encode the input as state vectors.
		states_value = self.encoder_model.predict(sequence)

		# Generate empty target sequence with first char == beginning 
		target_seq = np.zeros((1, _cfg.max_sequence_length))
		target_seq[0, 0] = 1 #self.reverse_word_idx[_cfg.beg_token] # shoudl be 1

		stop_condition = False
		decoded_sentence = ''
		while not stop_condition:
			output_tokens, h, c = self.decoder_model.predict(
				[target_seq] + states_value)

			# Sample a token
			sampled_token_index = np.argmax(output_tokens[0, -1, :])
			sampled_char = self.reverse_word_idx[sampled_token_index]
			decoded_sentence += sampled_char

			# Exit condition: either hit max length
			# or find stop character.
			if (sampled_char == 'eos' or
				len(decoded_sentence) > _cfg.max_sequence_length):
				stop_condition = True

			# Update the target sequence (of length 1).
			target_seq = np.zeros((1, _cfg.max_sequence_length))
			target_seq[0, 0] = sampled_token_index

			# Update states
			states_value = [h, c]

		return decoded_sentence


class Seq2SeqKeras:
	"""
	Keras implementation of Seq2Seq Model
	based on [link - Sutskever et al]
	"""
	def __init__(self, tokenizer, embed_weights=None):
		self.build(embed_weights)
		self.decoder = Decoder(self.encode_decode, tokenizer)

	def build(self, embed_weights=None):
		_cfg.num_tokens = np.shape(embed_weights)[0]
		## Set up encoder
		encoder_inputs = Input(shape=(_cfg.max_sequence_length,))
		x = Embedding(_cfg.num_tokens, _cfg.embed_size, name="emb_encoder")(encoder_inputs)
		x, state_h, state_c = LSTM(_cfg.latent_dim, return_state=True, name="encoder_lstm")(x)
		encoder_states = [state_h, state_c]

		## Set up the decoder, using `encoder_states` as initial state.
		decoder_inputs = Input(shape=(_cfg.max_sequence_length,))
		d = Embedding(_cfg.num_tokens, _cfg.embed_size, name="emb_decoder")(decoder_inputs)
		decoder_lstm = LSTM(_cfg.latent_dim, return_sequences=True, return_state=True, name="decoder_lstm")
		decoder_outputs, _, _ = decoder_lstm(d, initial_state=encoder_states)
		decoder_outputs = Dense(_cfg.num_tokens, activation='softmax', name="decoder_dense")(decoder_outputs)

		# def custom_objective(y_true, y_pred):
		# 	'''Just another crossentropy'''
		# 	loss = 0;

		# 	print(K.shape(y_true))
		# 	print(y_pred.get_shape())
		# 	print(K.shape(y_pred))
		# 	print(y_true)
		# 	print(K.gather(y_pred, ))
		# 	print(K.argmax(y_pred))
		# 	for i in range(len(y_true)):
		# 		gold = i[0]
		# 		pred = np.argmax(y_pred[i])
		# 		if gold == 0:
		# 			loss += 2 # penalty for padded values
		# 		elif gold != pred:
		# 			# motivate to increase probability of correc word 
		# 			# and decrease probability of incorrect word
		# 			# loss == probability of wrong value + (1- probability of correct value)
		# 			loss += y_pred[pred] + (1-y_pred[gold])
			# return loss
				

		# Main model that learns weights to predict target values
		self.encode_decode = Model([encoder_inputs, decoder_inputs], decoder_outputs)
		self.encode_decode.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		# self.encode_decode.compile(loss=custom_objective, optimizer='adadelta', metrics=['accuracy'])
		
		if embed_weights is not None:
			self.encode_decode.get_layer("emb_encoder").set_weights([embed_weights])
			self.encode_decode.get_layer("emb_decoder").set_weights([embed_weights])

		self.encode_decode.save('model/init.h5')
		print(self.encode_decode.summary())
	
	def eval_batch(self, batch):
		if batch % _cfg.eval_every == 0:
			self.decoder.decode_batch(queries, "chats/results_batch_{}.txt".format(batch))


	def train(self, inputs_train, target_train, inputs_val, target_val):
		# define callback - to save model and test decoding

		filepath="model/weights.hdf5" # this is weights, not full model
		checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
		decoder_callback = LambdaCallback(on_batch_end=lambda batch,logs: self.eval_batch(batch))
		callbacks_list = [checkpoint, decoder_callback]

		history = self.encode_decode.fit(inputs_train, target_train,
			batch_size=_cfg.batch_size, 
			epochs=_cfg.epochs, 
			callbacks=callbacks_list, 
			validation_data=(inputs_val, target_val),
			verbose=1)





