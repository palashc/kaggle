#https://towardsdatascience.com/multi-label-text-classification-with-scikit-learn-30714b7819c5
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

train_file_path = "/home/palash/kaggle/NLP/classification/toxic-comments/data/train.csv"
test_file_path = "/home/palash/kaggle/NLP/classification/toxic-comments/data/test.csv"

df = pd.read_csv(train_file_path, encoding = "ISO-8859-1")
df_test = pd.read_csv(test_file_path, encoding = "ISO-8859-1")
#print df.head()  #prints first 5 rows


###### Number of comments in each category #######################

df_toxic = df.drop(['id', 'comment_text'], axis=1)
counts = []
categories = list(df_toxic.columns.values)
for i in categories:
    counts.append((i, df_toxic[i].sum()))
df_stats = pd.DataFrame(counts, columns=['category', 'number_of_comments'])

#print df_stats
'''
df_stats.plot(x='category', y='number_of_comments', kind='bar', legend=False, grid=True, figsize=(8, 5))
plt.title("Number of comments per category")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('category', fontsize=12)
plt.show()
'''

##### Comments with multi-label #####################################

rowsums = df.iloc[:,2:].sum(axis=1)
x=rowsums.value_counts()
#plot
'''
plt.figure(figsize=(8,5))
ax = sns.barplot(x.index, x.values)
plt.title("Multiple categories per comment")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('# of categories', fontsize=12)
#plt.show()
'''

##### Number of words in comments ###################################

lens = df.comment_text.str.len()
lens.hist(bins = np.arange(0,5000,50))
#plt.show()

#### Cleaning #########################################################

#print df['comment_text'][0]
'''
Explanation
Why the edits made under my username Hardcore Metallica Fan were reverted? 
They weren't vandalisms, just closure on some GAs after I voted at New York Dolls FAC. 
And please don't remove the template from the talk page since I'm retired now.89.205.38.27
'''

def clean_text(text):
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
	text = re.sub(r"\'scuse", " excuse ", text)
	text = re.sub('\W', ' ', text)
	text = re.sub('\s+', ' ', text)
	text = text.strip(' ')
	return text

#clean text in dataframe
#print df['comment_text'][2]
df['comment_text'] = df['comment_text'].map(lambda comment : clean_text(comment))
df_test['comment_text'] = df_test['comment_text'].map(lambda comment : clean_text(comment))
#print df['comment_text'][2]

cleaned_train_file_path = "/home/palash/kaggle/NLP/classification/toxic-comments/data/cleaned_train.csv"
cleaned_test_file_path = "/home/palash/kaggle/NLP/classification/toxic-comments/data/cleaned_test.csv"
df.to_csv(cleaned_train_file_path, sep=',', encoding='utf-8')
df_test.to_csv(cleaned_test_file_path, sep=',', encoding='utf-8')

##### Done #######################