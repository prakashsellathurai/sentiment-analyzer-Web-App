# -*- coding: utf-8 -*-
from flask import Flask, request, render_template

import json
import logging
import os


from sklearn.externals import joblib

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(APP_ROOT, 'classifier.pkl')

PORT = 5000

app = Flask(__name__,template_folder='views')
logging.basicConfig(filename='movie_classifier.log', level=logging.DEBUG)
model = joblib.load(MODEL)
label = {0: 'negative', 1: 'positive'}


@app.route('/')
def home():
    return render_template('home.html')


def predict(model, text):
    return label[model.predict([text])[0]]


@app.route('/review', methods=['GET', 'POST'])
def extract():
    """Return the movie review sentiment score.

    Returns a JSON object :
    {
         "sentiment": "positive"
    }
    """
    if request.method == 'GET':
        description = request.args.get('text', '')

        result = {
            'sentiment': predict(model, description)
        }
        return json.dumps(result)
    elif request.method == 'POST':


        data = request.json

        description = (type(data['input-text']) !=  "undefined") and data['input-text'] or ""
        result = {
            'sentiment': predict(model, description)
        }
        print("proceesed")
        return json.dumps(result)

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=PORT, debug=True)
