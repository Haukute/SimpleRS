
import numpy as np
import pandas as pd
import keras
from keras.models import load_model

import tensorflow as tf

global graph
graph = tf.get_default_graph()


filepath = 'Data/ml-1m/u1.test'
df = pd.read_csv(filepath, sep = '\t', names = ['user', 'item', 'rating', 'time'])
df = df.drop(columns=['time'])
data = np.array(df)


def get_user(input_user, matrix):
    user_id = []
    for i in range(len(matrix)):
        if matrix[i][0] == input_user:
            user_id.append(matrix[i])
    user_id = np.array(user_id).reshape(-1,3)
    model = load_model('train_100k.h5')
    pre = model.predict([user_id[:,0], user_id[:,1]])
    result = np.concatenate((user_id,pre), axis=1)
    result = result[result[:,3].argsort()[::-1]]
    top_10 = result[:10]
    return top_10
def pre_dict(input_matrix):
    
    pre = model.predict([input_matrix[:,0], input_matrix[:,1]])
    input_matrix = np.concatenate((input_matrix,pre), axis=1)
    input_matrix = input_matrix[input_matrix[:,3].argsort()[::-1]]
    top_10 = input_matrix[:10]
    return top_10


def excute(matrix, user_id):
	kq = pre_dict(input_matrix=get_user(user_id,matrix))
	return kq
