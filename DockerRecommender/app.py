import pickle
import os
import json
import random
from pyspark import SparkContext, SQLContext
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import Row
from pyspark.sql.functions import *
from flask import Flask, request

app = Flask(__name__)

sc = SparkContext(appName="Yelp")
sc.setLogLevel("ERROR")

sqlc = SQLContext(sc)

with open('/webapp/Edinburgh_Restaurants_review.pickle', 'rb') as f:
	all_visited = pickle.load(f)
    
with open('/webapp/Edinburgh_Restaurants_business.pickle', 'rb') as f:
	rest = pickle.load(f)

with open('/webapp/Edinburgh_Restaurants_user.pickle', 'rb') as f:
	user = pickle.load(f)
    
n_business = len(rest)

if os.path.exists('./metastore_db/dbex.lck'):
	os.remove('./metastore_db/dbex.lck')
    
best_model = ALSModel.load('/webapp/Edinburgh_Restaurants.model')

if os.path.exists('./metastore_db/dbex.lck'):
	os.remove('./metastore_db/dbex.lck')

@app.route("/random_user", methods=["GET"])
def random_user():
	return random.choice(user.keys())

@app.route("/recommend", methods=["GET"])
def make_pred():
	user_id = request.args.get('user')
	try:
		n = int(request.args.get('n'))
	except (ValueError, TypeError):
		n = 5

	user_idn = user[user_id]
	visited = all_visited[user_idn]
	test_user = sqlc.createDataFrame([Row(user_idn=user_idn, business_idn=float(i)) for i in list(set(range(n_business)).difference(set(visited)))])

	pred_test = best_model.transform(test_user).na.fill(-5.0)
	top_pred = pred_test.orderBy(desc('prediction')).select('business_idn').rdd.map(lambda row: row.business_idn).take(n)
	response = map(lambda idn: rest[idn], top_pred)
	return json.dumps(response)

@app.route("/list", methods=["GET"])
def list_ratings():
	user_id = request.args.get('user')
	try:
		n = int(request.args.get('n'))
	except (ValueError, TypeError):
		n = 5

	user_idn = user[user_id]
	visited = all_visited[user_idn]
	response = sorted(map(lambda idn: rest[idn], visited), key=lambda k: k['stars'], reverse=True)[:n]
	return json.dumps(response)

if __name__ == "__main__":
    app.run(port=8001)
