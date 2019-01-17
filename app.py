
import recommend
import pandas as pd
import numpy as np
##---------------------------------------##
from flask import Flask, request, render_template
app = Flask(__name__)
##---------------------------------------##
import keras
from keras.models import load_model

import tensorflow as tf

global graph
graph = tf.get_default_graph()

# global model

model = load_model('train_100k.h5')
model._make_predict_function()
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
    pre = model.predict([user_id[:,0], user_id[:,1]])
    result = np.concatenate((user_id,pre), axis=1)
    result = result[result[:,3].argsort()[::-1]]
    top_10 = result[:10]
    return top_10


@app.route("/")
def index():
	return render_template('index.html')



@app.route("/result", methods=['GET', 'POST'])
def result():
	user_id = int(request.form.get('uname'))
	print('='*20)
	print(type(user_id))
	kq = get_user(user_id, data)
	return render_template('index.html', list_result= kq)


if __name__ == '__main__':
    app.run(debug=True)
