import pickle, os
import numpy as np

GLOVE_DIR = "/home/palash/kaggle/NLP/doc-similarity/data/"
EMBEDDING_DIM = 200

word_index = pickle.load(open("../data/word_index.pkl", "r"))

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.200d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


pickle.dump(embedding_matrix, open("../data/embedding_matrix.pkl", "w"))