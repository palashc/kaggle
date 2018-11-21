from keras.models import load_model
import pickle
import pandas as pd
import numpy as np

model = load_model('fasttext_like_model.h5')
x_test = pickle.load(open("../data/tokenized_test_data.pkl", "r"))
test_file_path = "/home/palash/kaggle/NLP/classification/toxic-comments/data/cleaned_test.csv"
df_test = pd.read_csv(test_file_path, encoding = "ISO-8859-1")
ids = df_test['id'].values

predictions = model.predict(x_test)

columns=['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
submission = pd.DataFrame(np.column_stack([ids, predictions]), columns=columns)

#print submission
submit_file_path = "/home/palash/kaggle/NLP/classification/toxic-comments/data/submission-2.csv"
submission.to_csv(submit_file_path, sep=',', encoding='utf-8', index = False)