# ---------------------------------------------
# Configuration settings for Seq2Seq model 
# ---------------------------------------------

# max length of context/response sequence
max_sequence_length = 40
# max size of vocabulary to use
num_tokens = 20000
# validation set size
part_val = 0.1
# dimensionality of word embeddings
embed_size = 300
# output dimensionality of LSTM
latent_dim = 256
# beginning and end tokens
beg_token = "begos"
end_token = "eos"
# Batch size for training
batch_size = 128  
# Number of epochs to train for
epochs = 10