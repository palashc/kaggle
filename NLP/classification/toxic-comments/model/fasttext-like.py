#################################################
import numpy as np
from keras.preprocessing import sequence
from keras.models import Model, Input
from keras.layers import Dense, Embedding, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.preprocessing.text import Tokenizer
from keras.optimizers import Adam
import pickle
#################################################

x_train, x_test, y_train, y_test = pickle.load(open("../data/tokenized_data_and_labels.pkl", "r"))

max_features = 20000  # number of words we want to keep
maxlen = 100  # max length of the comments in the model
batch_size = 64  # batch size for the model
embedding_dims = 100  # dimension of the hidden variable, i.e. the embedding dimension


#model - fasttext like
comment_input = Input((maxlen,))

#embedding layer which maps vocab indices into embedding_dims dimensions
comment_emb = Embedding(max_features, embedding_dims, input_length=maxlen, 
                        embeddings_initializer="uniform")(comment_input)

# we add a GlobalMaxPooling1D, which will extract features from the embeddings
# of all words in the comment
h = GlobalMaxPooling1D()(comment_emb)  #max pooling for multi-label use case. For sentiment classification, average/sum pooling is better.

#try with another hidden layer
fc1 = Dense(20, activation='relu')(h)
# We project onto a six-unit output layer, and squash it with a sigmoid:
output = Dense(6, activation='sigmoid')(fc1)

model = Model(inputs=comment_input, outputs=output)

model.compile(loss='binary_crossentropy',
              optimizer=Adam(0.01),
              metrics=['accuracy'])

model.summary()
# #testing
# model.fit(x_train, y_train, batch_size=batch_size, verbose=1, epochs=3, validation_split=0.1)
# score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
# print "Test Accuracy: ", score[1] # ~98%


#final
model.fit(np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test)), batch_size=batch_size, verbose=1, epochs=5, validation_split=0.1)


model.save('fasttext_like_model-fc1.h5') 

