from keras.models import load_model
import pickle
import pandas as pd
import numpy as np

# def exponent_neg_manhattan_distance(left, right):
#     return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

# model = load_model('malstm0.h5')
# model.save_weights('malstm0_w.h5')
# x_test = pickle.load(open("../data/tokenized_test_data.pkl", "r"))
# test_file_path = "/home/palash/kaggle/NLP/classification/toxic-comments/data/cleaned_test.csv"

# df_test = pd.read_csv(test_file_path, encoding = "ISO-8859-1")
# ids = df_test['test_id'].values

# predictions = model.predict(x_test)

# columns=['test_id', 'is_duplicate']
# submission = pd.DataFrame(np.column_stack([ids, predictions]), columns=columns)

# #print submission
# submit_file_path = "/home/palash/kaggle/NLP/classification/toxic-comments/data/submission-1.csv"
# submission.to_csv(submit_file_path, sep=',', encoding='utf-8', index = False)

submission1 = pd.read_csv("../data/submission-2.csv")
submission1.test_id = submission1.test_id.astype(int)
submission1.is_duplicate = submission1.is_duplicate.astype(float)

def predict(num):
	if num>0.5:
		return 1
	else:
		return 0


submission1['is_duplicate'] = submission1['is_duplicate'].map(lambda i: predict(i))

submission1.to_csv("../data/submission-fixed2.csv", sep=',', encoding='utf-8', index=False)
