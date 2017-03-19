import pickle
import os
import json
import random
import numpy as np
import tensorflow as tf
from flask import Flask, request
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

app = Flask(__name__)

features = ['cool', 'funny', 'stars', 'useful']

tokenizers = {}
models = {}
graphs = {}

for feature in features:
	with open('/webapp/' + feature + '/' + feature + '.tokenizer', 'rb') as f:
		tokenizers.update({feature: pickle.load(f)})

	models.update({feature: load_model('/webapp/' + feature + '/' + feature + '.model')})
	# https://github.com/fchollet/keras/issues/2397
	graphs.update({feature: tf.get_default_graph()})

def make_pred(feature, sentence):
	sequence = tokenizers[feature].texts_to_sequences([sentence])
	padded_seq = pad_sequences(sequence, maxlen=400)
	with graphs[feature].as_default():
		predicted_proba = models[feature].predict_proba(padded_seq)
	return predicted_proba

@app.route("/cool", methods=["POST"])
def cool():
	sentence = request.form.get('review').encode('utf-8')
	predicted_proba = make_pred('cool', sentence)
	prediction = (predicted_proba > 0.5).astype(int)[0][0]
	response = {'cool': prediction}
	return json.dumps(response)

@app.route("/funny", methods=["POST"])
def funny():
	sentence = request.form.get('review').encode('utf-8')
	predicted_proba = make_pred('funny', sentence)
	prediction = (predicted_proba > 0.5).astype(int)[0][0]
	response = {'funny': prediction}
	return json.dumps(response)

@app.route("/stars", methods=["POST"])
def stars():
	sentence = request.form.get('review').encode('utf-8')
	predicted_proba = make_pred('stars', sentence)
	prediction = np.argmax(predicted_proba, axis=1)[0]
	response = {'stars': prediction}
	return json.dumps(response)

@app.route("/useful", methods=["POST"])
def useful():
	sentence = request.form.get('review').encode('utf-8')
	predicted_proba = make_pred('useful', sentence)
	prediction = (predicted_proba > 0.5).astype(int)[0][0]
	response = {'useful': prediction}
	return json.dumps(response)

if __name__ == "__main__":
    app.run(port=8002)
