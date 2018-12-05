import re, pickle
import config as _cfg
import numpy as np


class progressBar:
    def __init__(self, total, prefix = '', suffix = '', length = 100, fill = '█'):
        self.percent = 0
        self.iteration = 0
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.length = length
        self.fill = fill
        self.print()

    def update(self, iteration):
        self.iteration = iteration
        self.percent = ("{0:.1f}").format(100 * (iteration / float(self.total)))
        self.print()
    
    def print(self):
        filledLength = int(self.length * self.iteration // self.total)
        bar = self.fill * filledLength + '-' * (self.length - filledLength)
        print('\r%s |%s| %s%% %s' % (self.prefix, bar, self.percent, self.suffix), end = '\r')

        if self.iteration == self.total: 
            print()

def clean_word(word):
    if word.endswith("n'"):
        word = "{}g".format(word[0:-1])
    return word

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\.\'\`-]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\.\.\.", " ... ", string)
    string = re.sub(r"-{2,}", " -- ", string)
    string = re.sub(r"\s{2,}", " ", string)
    
    string = string.strip().lower()
    words = [clean_word(w) for w in string.split()]
    string = " ".join(words)
    
    return string

def line_num(s):
    return int(re.sub('\D', '', s))

def save_var(obj, file):
    print("saving to {}".format(file))
    with open(file, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def text_to_seq(text, tokenizer, seq_len):
    text = clean_str(text)
    tok = tokenizer.texts_to_sequences(["<begos> {} <eos>".format(_cfg.beg_token, text, _cfg.end_token)])
    padded = pad_sequences(tok, maxlen=_cfg.max_sequence_length, padding='pre', value=0)
    return padded

def load_embeddings(source, word_index, max_features):
    """
    Loads pre-trained embeddings and returns embedding matrix 
    for max_features words used in the model
    Taken from https://www.kaggle.com/shujian/different-embeddings-with-attention-fork-fork/notebook
    """
    
    print("Loading embeddings from {}".format(source))

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(source))

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    nb_words = min(max_features, len(word_index)+1)
    # TODO update config num_tokens..
    
    # initialize embedding weights to random
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    
    # update weights if found in loaded embeddings
    num_loaded = 0
    not_found = []
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        
        ## lower case?
        if embedding_vector is None:
            embedding_vector = embeddings_index.get(word.lower())
        
        ## missing a g at the end? eg talkin' -> talking
        if embedding_vector is None and word.endswith("n'"):
            embedding_vector = embeddings_index.get("{}g".format(word.lower()[0:-1]))
            
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector
            num_loaded += 1
        else:
            not_found.append(word)
        
    
    print("Loaded embeddings cover {:.2f}% of tokens in training data".format(100*num_loaded/nb_words))
    return embedding_matrix, not_found