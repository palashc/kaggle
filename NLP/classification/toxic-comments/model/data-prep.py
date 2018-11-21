#https://www.depends-on-the-definition.com/classify-toxic-comments-on-wikipedia/
#### Imports ###############################
#%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from sklearn.model_selection import train_test_split

from keras.preprocessing import sequence
from keras.models import Model, Input
from keras.layers import Dense, Embedding, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.preprocessing.text import Tokenizer
from keras.optimizers import Adam
import pickle
#### Train/Test split #########################################
train_file_path = "/home/palash/kaggle/NLP/classification/toxic-comments/data/cleaned_train.csv"
test_file_path = "/home/palash/kaggle/NLP/classification/toxic-comments/data/cleaned_test.csv"

df = pd.read_csv(train_file_path, encoding = "ISO-8859-1")
df_test = pd.read_csv(test_file_path, encoding = "ISO-8859-1")
df_test.comment_text = df_test.comment_text.astype(str)

categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train, test = train_test_split(df, random_state=42, test_size=0.2, shuffle=True)

X_train = train['comment_text'].values
X_test = test['comment_text'].values
X_final_test = df_test['comment_text'].values

y_train = train[categories].values
y_test = test[categories].values
#### Model ############################################

max_features = 20000  # number of words we want to keep
maxlen = 100  # max length of the comments in the model
batch_size = 64  # batch size for the model
embedding_dims = 100  # dimension of the hidden variable, i.e. the embedding dimension

# bag of Words
tok = Tokenizer(num_words=max_features)
tok.fit_on_texts(list(X_train) + list(X_test))
x_train = tok.texts_to_sequences(X_train)
x_test = tok.texts_to_sequences(X_test)
x_final_test = tok.texts_to_sequences(X_final_test)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))

# padding
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
x_final_test = sequence.pad_sequences(x_final_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

data = (x_train, x_test, y_train, y_test)
pickle.dump(data, open("../data/tokenized_data_and_labels-emb100.pkl", "w"))
pickle.dump(x_final_test, open("../data/tokenized_test_data-emb100.pkl", "w"))
