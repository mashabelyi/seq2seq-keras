#!/usr/bin/env python3
import argparse, pickle
import config as _cfg
from utils import *
from seq2seq import Seq2SeqKeras
from data_loader import DataLoader


def main():
	parser = argparse.ArgumentParser(description='Seq2sec model configuration arguments')

	parser.add_argument('--input', type=str, 
    	required=True, help='training data file.')
	parser.add_argument('--pre_embed', type=str,
    	help='pre-trained embeddings file to use as input for model.')

	args = parser.parse_args()
	
	# load and process data
	loader = DataLoader(args.input)
	in_train, target_train = loader.get_training()
	in_val, target_val = loader.get_val()


	# load emebddings
	embed_weights = None
	if args.pre_embed is not None:
		embed_weights, _ = load_embeddings(args.pre_embed, loader.tokenizer.word_index, _cfg.num_tokens)

		#save the loaded embeddings
		print("saving initialized embedding weights to {}".format('model/embed_weights_init.pkl'))
		with open('model/embed_weights_init.pkl', 'wb') as f:
			pickle.dump(embed_weights, f, pickle.HIGHEST_PROTOCOL)

	# train
	model = Seq2SeqKeras(loader.tokenizer, embed_weights)

	model.train(
		in_train, target_train,
		in_val, target_val)

if __name__ == '__main__':
	main()