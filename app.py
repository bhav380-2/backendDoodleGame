from flask import Flask, request, jsonify
import json
import pandas as pd
import numpy as np
from tensorflow import keras
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
file_path = './categories.json' 

with open(file_path, 'r') as file:
    data = json.load(file) 
    classes_path = data['categories']
classes_path = sorted(classes_path, key=lambda s: s.lower())
class_dict = {x[:-4].replace(" ", "_"):i for i, x in enumerate(classes_path)}
path  = "../dataset/train_simplified/train_simplified"



def load_model():
    model = keras.models.load_model('CNN1.keras')
    return model
model = load_model()

@app.route('/',methods=['GET'])
def home():
    print("xxx")
    return {"success":True}

@app.route('/predict', methods=['POST'])
def predict():
    print("HIIIIII")
    data = request.get_json(force=True)
    print(data)
    image = np.vstack(data).reshape(64,64)
    image = image.reshape(1,64,64,1)
    y = model.predict(image,verbose=1)
    top_3 =  np.argsort(-y)[:, 0:6]
    pred_results = []
    pred_results.append(top_3)
    pred_results = np.concatenate(pred_results)
    reverse_dict = {v: k for k, v in class_dict.items()}
    preds_df = pd.DataFrame({'first': pred_results[:,0], 'second': pred_results[:,1], 'third': pred_results[:,2]})
    preds_df = preds_df.replace(reverse_dict)

    result = []
    result.append(preds_df['first'].iloc[0])
    result.append(preds_df['second'].iloc[0])
    result.append(preds_df['third'].iloc[0])
    # print(result)
    print(jsonify(result))
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
