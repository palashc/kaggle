#################################################
import numpy as np
from keras.preprocessing import sequence
from keras.models import Model, Input
from keras.layers import Dense, Embedding, LSTM, Lambda
from keras.preprocessing.text import Tokenizer
from keras.optimizers import Adam
import pickle
import keras.backend as K
import pandas as pd
#################################################

x_train, x_test, y_train, y_test = pickle.load(open("../data/tokenized_data_and_labels.pkl", "r"))

max_features = 20000  # number of words we want to keep
maxlen = 100  # max length of the comments in the model
batch_size = 64  # batch size for the model
embedding_dims = 200  # dimension of the hidden variable, i.e. the embedding dimension
n_hidden = 50 # number of hidden lstm units
gradient_clipping_norm = 1.25
n_epoch = 1
### Model

#input
left_input = Input((maxlen,))
right_input = Input((maxlen,))

#embeddings
embedding_layer = Embedding(max_features, embedding_dims, input_length=maxlen, embeddings_initializer="uniform")
left_emb = embedding_layer(left_input)
right_emb = embedding_layer(right_input)

#lstm
shared_lstm = LSTM(n_hidden)
left_output = shared_lstm(left_emb)
right_output = shared_lstm(right_emb)

#malstm
def exponent_neg_manhattan_distance(left, right):
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

# Calculates the distance as defined by the MaLSTM model
malstm_distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

# Pack it all up into a model
malstm = Model([left_input, right_input], [malstm_distance])

#optimizer = Adam(0.01, clipnorm=gradient_clipping_norm)
#malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

malstm.summary()

# trained_model = malstm.fit([x_train['left'], x_train['right']], y_train, batch_size=batch_size, epochs=n_epoch, validation_split=0.1)
# malstm.save_weights('malstm0_w.h5') 
# ## testing
# score = malstm.evaluate([x_test['left'], x_test['right']], y_test, batch_size=batch_size, verbose=1)
# print "Test Accuracy: ", score[1]

# #Plot accuracy
# # plt.plot(trained_model.history['acc'])
# # plt.plot(trained_model.history['val_acc'])
# # plt.title('Model Accuracy')
# # plt.ylabel('Accuracy')
# # plt.xlabel('Epoch')
# # plt.legend(['Train', 'Validation'], loc='upper left')
# # plt.show()

# # # Plot loss
# # plt.plot(trained_model.history['loss'])
# # plt.plot(trained_model.history['val_loss'])
# # plt.title('Model Loss')
# # plt.ylabel('Loss')
# # plt.xlabel('Epoch')
# # plt.legend(['Train', 'Validation'], loc='upper right')
# # plt.show()

## submission

malstm.load_weights('malstm0_w.h5')
x_test = pickle.load(open("../data/tokenized_test_data.pkl", "r"))
test_file_path = "/home/palash/kaggle/NLP/doc-similarity/data/cleaned_test.csv"

df_test = pd.read_csv(test_file_path, encoding = "ISO-8859-1")
print df_test.head()
ids = df_test['test_id'].values

predictions = malstm.predict([x_test['left'], x_test['right']], batch_size=batch_size, verbose=1)

columns=['test_id', 'is_duplicate']
submission = pd.DataFrame(np.column_stack([ids, predictions]), columns=columns)

#print submission
submit_file_path = "/home/palash/kaggle/NLP/doc-similarity/data/submission-1.csv"
submission.to_csv(submit_file_path, sep=',', encoding='utf-8', index = False)

