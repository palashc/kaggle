#################### Imports ###################################
#%matplotlib inline
import re
import glob
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import seaborn as sns
import pickle
##################################################################

train_folder = "/home/palash/kaggle/NLP/transfer-learning/data/train/"
test_file_path = "/home/palash/kaggle/NLP/transfer-learning/data/test.csv"
categories = ['biology', 'cooking', 'crypto', 'diy', 'robotics', 'travel']

allFiles = glob.glob(train_folder + "/*.csv")

df = pd.DataFrame()
list_ = []
for file in allFiles:
    df = pd.read_csv(file,index_col=None, header=0)
    list_.append(df)
df = pd.concat(list_)
df_test = pd.read_csv(test_file_path, encoding = "ISO-8859-1")
#print df['content'].values[0]

#### text cleaning ##################################################

def clean(text, content=False):
	text = text.lower()
	text = re.sub('\s+', ' ', text)
	if content:
		text = re.sub(r'<.*?>', '', text)
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
		text = text.strip(' ')
		words = text.split()
		new_words = [word for word in words if word not in stop_words]
		text = " ".join(new_words)
	return text

#train data
print "Cleaning training data"
df['title'] = df['title'].map(lambda title : clean(title, True))
df['content'] = df['content'].map(lambda content : clean(content, True))
df['tags'] = df['tags'].map(lambda tags : clean(tags))
cleaned_train_file_path = "/home/palash/kaggle/NLP/transfer-learning/data/cleaned_train.csv"
df.to_csv(cleaned_train_file_path, sep=',', encoding='utf-8', index=False)

#test data
print "Cleaning testing data"
df_test['title'] = df_test['title'].map(lambda title : clean(title, True))
df_test['content'] = df_test['content'].map(lambda content : clean(content, True))
cleaned_test_file_path = "/home/palash/kaggle/NLP/transfer-learning/data/cleaned_test.csv"
df.to_csv(cleaned_test_file_path, sep=',', encoding='utf-8', index=False)

#################storing list of unique tags############################################################
print "getting list of tags"

def get_tags(tags):
	tags = tags.lower()
	tags = re.sub('\s+', ' ', tags)
	tags = tags.split()
	return tags

list_ = []
for tag_set in df['tags'].map(lambda tags : get_tags(tags)):
	list_.extend(tag_set)

tags = pd.DataFrame(list_, columns = ['tags'])
#print len(set(tags.tags.values))

unique_tags = set(tags.tags.values)

tag_dict = dict(enumerate(unique_tags))
pickle.dump(tag_dict, open("../data/tag_dict.pkl","w"))

################################################################################################