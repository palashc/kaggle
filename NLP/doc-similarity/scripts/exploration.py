#################### Imports ###################################
#%matplotlib inline
import re
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import seaborn as sns
##################################################################

#train_file_path = "/home/palash/kaggle/NLP/doc-similarity/data/train.csv"
test_file_path = "/home/palash/kaggle/NLP/doc-similarity/data/fixed_test.csv"

#df_train = pd.read_csv(train_file_path, low_memory=False)
df_test = pd.read_csv(test_file_path, low_memory=False)

## clean data

def clean(text):
	text = str(text)
	text = text.lower()
	text = re.sub(r"what's", "what is ", text)
	text = re.sub(r"\'s", " ", text)
	text = re.sub(r"\'ve", " have ", text)
	text = re.sub(r"can't", "can not ", text)
	text = re.sub(r"n't", " not ", text)
	text = re.sub(r"i'm", "i am ", text)
	text = re.sub(r"\'re", " are ", text)
	text = re.sub(r"\'d", " would ", text)
	text = re.sub(r"\'ll", " will ", text)
	text = re.sub('\W', ' ', text)
	text = re.sub('\s+', ' ', text)
	text = text.strip(' ')
	words = text.split()
	new_words = [word for word in words if word not in stop_words]
	text = " ".join(new_words)
	return text

#df_train['question1'] = df_train['question1'].map(lambda q: clean(q))
#df_train['question2'] = df_train['question2'].map(lambda q: clean(q))

df_test['question1'] = df_test['question1'].map(lambda q: clean(q))
df_test['question2'] = df_test['question2'].map(lambda q: clean(q))

#cleaned_train_file_path = "/home/palash/kaggle/NLP/doc-similarity/data/cleaned_train.csv"
cleaned_test_file_path = "/home/palash/kaggle/NLP/doc-similarity/data/cleaned_test.csv"
#df_train.to_csv(cleaned_train_file_path, sep=',', encoding='utf-8')
df_test.to_csv(cleaned_test_file_path, sep=',', encoding='utf-8')