import flask, pickle, json, os, time
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

#testing
#testing some more
#testing even more
#the next day...

lctime = time.localtime(os.path.getmtime("MNB.pickle"))
app.date = time.strftime("%Y-%m-%d %H:%M:%S", lctime)

app.classifier = pickle.load(open("MNB.pickle", "rb"))
app.tfidf = pickle.load(open("VCTer.pickle", "rb"))

@app.route("/api/american", methods = ["POST"])

def prediction():
    inp = request.get_json(force = True)
    text = np.array([inp['text']])
    text = app.tfidf.transform(text)
    opt = app.classifier.predict(text)
    return jsonify({"is_american": str(opt[0]), "version": "MultinomialNB_v1", "model_date": app.date})

if __name__ == '__main__':
    app.run(port = 5010, host = '0.0.0.0')

'''
wget --server-response --output-document response.out --header='Content-Type: application/json' --post-data '{"text": "I am American"}' http://localhost:5010/api/american
'''
