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
#################################################

#train/test split

train_file_path = "/home/palash/kaggle/NLP/doc-similarity/data/cleaned_train.csv"
test_file_path = "/home/palash/kaggle/NLP/doc-similarity/data/cleaned_test.csv"

df_train = pd.read_csv(train_file_path, low_memory=False)
df_train.question1 = df_train.question1.astype(str)
df_train.question2 = df_train.question2.astype(str)

df_test = pd.read_csv(test_file_path, low_memory=False)
df_test.question1 = df_test.question1.astype(str)
df_test.question2 = df_test.question2.astype(str)

train, test = train_test_split(df_train, random_state=42, test_size=0.1, shuffle=True)

X_train = {}
X_train['left'] = train['question1'].values
X_train['right'] = train['question2'].values

X_test = {}
X_test['left'] = test['question1'].values
X_test['right'] = test['question2'].values

X_final_test = {}
X_final_test['left'] = df_test['question1'].values
X_final_test['right'] = df_test['question2'].values

y_train = train['is_duplicate'].values
y_test = test['is_duplicate'].values

#################################################################

# tokenize

max_features = 20000  # number of words we want to keep
maxlen = 100  # max length of the comments in the model

tok = Tokenizer(num_words=max_features)
tok.fit_on_texts(list(X_train['left']) + list(X_train['right']) + list(X_test['left']) + list(X_test['right']))
pickle.dump(tok, open("../data/tokenizer.pkl","w"))
del X_train
del X_test
del df_train
del df_test
x_train, x_test, x_final_test = {}, {}, {}
# x_train['left'] = tok.texts_to_sequences(X_train['left'])
# x_train['right'] = tok.texts_to_sequences(X_train['right'])
# x_test['left'] = tok.texts_to_sequences(X_test['left'])
# x_test['right'] = tok.texts_to_sequences(X_test['right'])

x_final_test['left'] = tok.texts_to_sequences(X_final_test['left'])
x_final_test['right'] = tok.texts_to_sequences(X_final_test['right'])

#word index
word_index = tok.word_index
print('Found %s unique tokens.' % len(word_index))

# print(len(x_train['left']), 'train pairs')
# print(len(x_test['left']), 'test pairs')
# print('Average train left length: {}'.format(np.max(list(map(len, x_train['left'])))))
# print('Average train right length: {}'.format(np.max(list(map(len, x_train['right'])))))
# print('Average test left length: {}'.format(np.max(list(map(len, x_test['left'])))))
# print('Average test right length: {}'.format(np.max(list(map(len, x_train['right'])))))

## padding

# x_train['left'] = sequence.pad_sequences(x_train['left'], maxlen=maxlen)
# x_train['right'] = sequence.pad_sequences(x_train['right'], maxlen=maxlen)
# x_test['left'] = sequence.pad_sequences(x_test['left'], maxlen=maxlen)
# x_test['right'] = sequence.pad_sequences(x_test['right'], maxlen=maxlen)

x_final_test['left'] = sequence.pad_sequences(x_final_test['left'], maxlen=maxlen)
x_final_test['right'] = sequence.pad_sequences(x_final_test['right'], maxlen=maxlen)

# print('x_train shape:', x_train['left'].shape)
# print('x_test shape:', x_test['left'].shape)

# print x_train['left'][0]

# data = (x_train, x_test, y_train, y_test)
# pickle.dump(data, open("../data/tokenized_data_and_labels.pkl", "w"))
#pickle.dump(word_index, open("../data/word_index.pkl", "w"))
pickle.dump(x_final_test, open("../data/tokenized_test_data.pkl", "w"))