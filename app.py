from flask import Flask, request
from flask_cors import CORS, cross_origin
import joblib
import numpy as np
import re


app = Flask(__name__)
CORS(app)

@app.route('/')
def helloworld():
    return 'Hello world'

@app.route('/fever', methods=['POST'])
@cross_origin()
def predict_species():
    model = joblib.load('fever.model')
    req = request.values['param']
    inputs = np.array(req.split(','), dtype=np.float32).reshape(1, -1)
    predict_target = model.predict(inputs)
    if predict_target == 0:
        return 'ไข้หวัดธรรมดา'
    elif predict_target == 1:
        return 'ไข้หวัดใหญ่'
    else:
        return 'ไข้เลือดออก'           

if __name__== '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)        