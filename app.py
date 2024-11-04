from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import keras 
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import LeakyReLU
import os

import json
import pandas as pd
# Path to your JSON file
app = Flask(__name__)
CORS(app)
file_path = './categories.json' 

# Open and read the JSON file
with open(file_path, 'r') as file:
    data = json.load(file) 
    classes_path = data['categories']

classes_path = sorted(classes_path, key=lambda s: s.lower())
class_dict = {x[:-4].replace(" ", "_"):i for i, x in enumerate(classes_path)}
Base_Size = 256
num_csv = 100
num_class = 340
size = 64
steps = 17578
batchsize = 256
epochs = 100
path  = "../dataset/train_simplified/train_simplified"


def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


def load_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(size,size,1),padding='same'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))                  
    model.add(Dense(num_class, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=0.01),
                        metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy])
    model.load_weights('./CNN.h5')
    return model

model = load_model()

@app.route('/predict', methods=['POST'])

def predict():
    data = request.get_json(force=True)
    image = np.vstack(data).reshape(64,64)
    image = image.reshape(1,64,64,1)
    y = model.predict(image,verbose=1)
    top_3 =  np.argsort(-y)[:, 0:6]
    pred_results = []
    pred_results.append(top_3)
    pred_results = np.concatenate(pred_results)
    reverse_dict = {v: k for k, v in class_dict.items()}
    preds_df = pd.DataFrame({'first': pred_results[:,0], 'second': pred_results[:,1], 'third': pred_results[:,2], '4':pred_results[:,3],'5':pred_results[:,4],'6':pred_results[:,5]})
    preds_df = preds_df.replace(reverse_dict)

    result = []
    result.append(preds_df['first'].iloc[0])
    result.append(preds_df['second'].iloc[0])
    result.append(preds_df['third'].iloc[0])
    # print(result)
    return jsonify(result)
if __name__ == '__main__':
    app.run(debug=True)
