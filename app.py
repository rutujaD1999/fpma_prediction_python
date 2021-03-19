
from datetime import datetime
import numpy as np
from matplotlib import cm, pyplot as plt
import pandas as pd
from pandas_datareader import data
import seaborn as sns
from sklearn import tree
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from flask import Flask, Response, json, request, jsonify
from flask_cors import CORS, cross_origin
from flask_restful import Resource, Api, reqparse
from firebase_admin import credentials, firestore, initialize_app
from apscheduler.scheduler import Scheduler

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
# api = Api(app)
request_queue = [] 
cred = credentials.Certificate('fpma_key.json')
default_app = initialize_app(cred)
db = firestore.client()
res = []
cron = Scheduler(daemon=True)
cron.start()
# class hello(Resource):
def prediction_queue(request_queue1):

	try:
		rq = request_queue1.pop(0)
		uid = rq['uid']
		st_name = rq['st_name']
		start_date = rq['start']
		end_date = rq['end']
		duration = 1
		sd = start_date.split()
		start = datetime(int(sd[0]), int(sd[1]), int(sd[2]))
		ed = end_date.split()
		end = datetime(int(ed[0]), int(ed[1]), int(ed[2]))
		df = data.DataReader(st_name, start=start, end=end, data_source='yahoo')
		df.to_csv("st_name.csv")
		df=pd.read_csv("st_name.csv")
		wR = ta.momentum.WilliamsRIndicator(df["High"], df["Low"], df["Close"]).williams_r()
		rsi = ta.momentum.rsi(df["Close"])
		mfi = ta.volume.money_flow_index(df["High"], df["Low"], df["Close"], df["Volume"])
		indicator_bb = ta.volatility.BollingerBands(df["Close"])
		upper = indicator_bb.bollinger_hband()
		middle = indicator_bb.bollinger_mavg()
		lower = indicator_bb.bollinger_lband()
		indicator_macd = ta.trend.MACD(df['Close'])
		macd = indicator_macd.macd()
		macdsignal = indicator_macd.macd_signal()
		macdhist = indicator_macd.macd_diff()
		df = df.assign(rsi = rsi)
		df = df.assign(mfi = mfi)
		df = df.assign(willR = wR)
		df = df.assign(bb_upper = upper)
		df = df.assign(bb_middle = middle)
		df = df.assign(bb_lower = lower)
		df = df.assign(macd = macd)
		df = df.assign(macdhist = macdhist)
		df = df.assign(macdsignal = macdsignal)
		df["ho"]=(((df.High-df.Open)/df.Open)*100)
		df["ol"]=(((df.Open-df.Low)/df.Open)*100)
		df["oc"]=(((df.Open-df.Close)/df.Open)*100)
		df["hl"]=(((df.High-df.Low)/df.Low)*100)
		df["pvc"]=(df["Volume"]-df["Volume"].shift(1))/df["Volume"].shift(1)*100
		df["ho1"]=df["ho"].shift(1)
		df["ol1"]=df["ol"].shift(1)
		df["oc1"]=df["oc"].shift(1)
		df["hl1"]=df["hl"].shift(1)
		df["pvc1"]=df["pvc"].shift(1)
		df["ho2"]=df["ho"].shift(2)
		df["ol2"]=df["ol"].shift(2)
		df["oc2"]=df["oc"].shift(2)
		df["hl2"]=df["hl"].shift(2)
		df["pvc2"]=df["pvc"].shift(2)
		df["ho3"]=df["ho"].shift(3)
		df["ol3"]=df["ol"].shift(3)
		df["oc3"]=df["oc"].shift(3)
		df["hl3"]=df["hl"].shift(3)
		df["pvc3"]=df["pvc"].shift(3)
		df["ho4"]=df["ho"].shift(4)
		df["ol4"]=df["ol"].shift(4)
		df["oc4"]=df["oc"].shift(4)
		df["hl4"]=df["hl"].shift(4)
		df["pvc4"]=df["pvc"].shift(4)
		df["ho5"]=df["ho"].shift(5)
		df["ol5"]=df["ol"].shift(5)
		df["oc5"]=df["oc"].shift(5)
		df["hl5"]=df["hl"].shift(5)
		df["pvc5"]=df["pvc"].shift(5)
		df["ho6"]=df["ho"].shift(6)
		df["ol6"]=df["ol"].shift(6)
		df["oc6"]=df["oc"].shift(6)
		df["hl6"]=df["hl"].shift(6)
		df["pvc6"]=df["pvc"].shift(6)
		df['Date']= pd.to_datetime(df["Date"])
		df['weekday'] = df['Date'].dt.dayofweek
		df["poc"]=(df["Close"].shift(-(duration-1))-df["Open"])/df["Open"]*100
		df["nextOpen"]=(df["Open"]-df["Close"].shift(1))/df["Close"].shift(1)*100
		df["Signal"]=""
		df['MA5']=df['Close'].rolling(window=5).mean()
		df['MA10']=df['Close'].rolling(window=10).mean()
		df['MA15']=df['Close'].rolling(window=15).mean()
		df['MA20']=df['Close'].rolling(window=20).mean()
		df['MA25']=df['Close'].rolling(window=25).mean()
		df['MA30']=df['Close'].rolling(window=30).mean()
		df['1MA5']=df['MA5'].shift(1)
		df['1MA10']=df['MA10'].shift(1)
		df['1MA15']=df['MA15'].shift(1)
		df['1MA20']=df['MA20'].shift(1)
		df['1MA25']=df['MA25'].shift(1)
		df['1MA30']=df['MA30'].shift(1)
		df['EMA5'] = df['Close'].ewm(span=5, adjust=False).mean()
		df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
		df['EMA15'] = df['Close'].ewm(span=15, adjust=False).mean()
		df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
		df['EMA25'] = df['Close'].ewm(span=25, adjust=False).mean()
		df['EMA30'] = df['Close'].ewm(span=30, adjust=False).mean()
		df['1EMA5']=df['EMA5'].shift(1)
		df['1EMA10']=df['EMA10'].shift(1)
		df['1EMA15']=df['EMA15'].shift(1)
		df['1EMA20']=df['EMA20'].shift(1)
		df['1EMA25']=df['EMA25'].shift(1)
		df['1EMA30']=df['EMA30'].shift(1)
		df["poc"].value_counts(bins=10,normalize=True)
		df["Signal"]=np.where(((df['poc'] >= 0.2)),"BUY",(np.where(((df['poc'] <= -0.2)),"SELL","HOLD")))
		df["Signal"].value_counts()
		df = df.replace([np.inf, -np.inf], np.nan)
		df=df.fillna(method='ffill')
		df[df.isnull().any(axis=1)]
		split=int(len(df)*0.97)
		a=np.where(df[['rsi','mfi', 'willR', 'bb_upper', 'bb_middle', 'bb_lower','macd', 'macdhist',
			'macdsignal','ho1', 'ol1', 'oc1', 'hl1', 'pvc1', 'ho2',
			'ol2', 'oc2', 'hl2', 'pvc2', 'ho3', 'ol3', 'oc3', 'hl3', 'pvc3', 'ho4',
			'ol4', 'oc4', 'hl4', 'pvc4', 'ho5', 'ol5', 'oc5', 'hl5', 'pvc5', 'ho6',
			'ol6', 'oc6', 'hl6', 'pvc6', 'weekday', 'poc', 'nextOpen', '1MA5', '1MA10', '1MA15',
			'1MA20', '1MA25', '1MA30', '1EMA5', '1EMA10', '1EMA15', '1EMA20', '1EMA25', '1EMA30']].values >= np.finfo(np.float64).max)
		aa=a[0]
		X_train = df[['rsi',
			'mfi', 'willR', 'bb_upper', 'bb_middle', 'bb_lower','macd', 'macdhist',
			'macdsignal','ho1', 'ol1', 'oc1', 'hl1', 'pvc1', 'ho2',
			'ol2', 'oc2', 'hl2', 'pvc2', 'ho3', 'ol3', 'oc3', 'hl3', 'pvc3', 'ho4',
			'ol4', 'oc4', 'hl4', 'pvc4', 'ho5', 'ol5', 'oc5', 'hl5', 'pvc5', 'ho6',
			'ol6', 'oc6', 'hl6', 'pvc6', 'weekday', 'nextOpen', '1MA5', '1MA10', '1MA15',
			'1MA20', '1MA25', '1MA30', '1EMA5', '1EMA10', '1EMA15', '1EMA20', '1EMA25', '1EMA30']][33:split]
		X_test = df[['rsi',
			'mfi', 'willR', 'bb_upper', 'bb_middle', 'bb_lower','macd', 'macdhist',
			'macdsignal','ho1', 'ol1', 'oc1', 'hl1', 'pvc1', 'ho2',
			'ol2', 'oc2', 'hl2', 'pvc2', 'ho3', 'ol3', 'oc3', 'hl3', 'pvc3', 'ho4',
			'ol4', 'oc4', 'hl4', 'pvc4', 'ho5', 'ol5', 'oc5', 'hl5', 'pvc5', 'ho6',
			'ol6', 'oc6', 'hl6', 'pvc6', 'weekday', 'nextOpen', '1MA5', '1MA10', '1MA15',
			'1MA20', '1MA25', '1MA30', '1EMA5', '1EMA10', '1EMA15', '1EMA20', '1EMA25', '1EMA30']][split:-2]
		Y_train = df[['Signal']][33:split]
		Y_test = df[['Signal']][split:-2]
		clf = RandomForestClassifier(n_estimators=1000,random_state=25)
		clf.fit(X_train, Y_train)
		p=clf.predict(X_train)
		# import pickle
		# pickle.dump(clf, open(‘model.pkl’, ‘wb’))
		np.unique(p)
		# Python script for confusion matrix creation. 
		
		
		t=clf.predict(X_test)
		np.unique(t)
		

		n=df[['rsi',
			'mfi', 'willR', 'bb_upper', 'bb_middle', 'bb_lower','macd', 'macdhist',
			'macdsignal','ho1', 'ol1', 'oc1', 'hl1', 'pvc1', 'ho2',
			'ol2', 'oc2', 'hl2', 'pvc2', 'ho3', 'ol3', 'oc3', 'hl3', 'pvc3', 'ho4',
			'ol4', 'oc4', 'hl4', 'pvc4', 'ho5', 'ol5', 'oc5', 'hl5', 'pvc5', 'ho6',
			'ol6', 'oc6', 'hl6', 'pvc6', 'weekday', 'nextOpen', '1MA5', '1MA10', '1MA15',
			'1MA20', '1MA25', '1MA30', '1EMA5', '1EMA10', '1EMA15', '1EMA20', '1EMA25', '1EMA30']][-1:]
		t1=clf.predict(n)
		print(t1[0])
		doc_ref = db.collection('users').document(uid)

		doc = doc_ref.get()
		datamap = doc.to_dict()
		checking = datamap.get('prediction_data')

		print("checking")
		print("after datamap\n")
		if 'prediction_data' in datamap:
			data_entry = datamap['prediction_data']
			datamap_entry = {
					'stock_name' : st_name,
					'start_date' : start,
					'end_date' : end,
					'prediction' : t1[0]
				}
			data_entry.append(datamap_entry)
			db.collection('users').document(uid).update({'prediction_data' : data_entry})
		else:
			data_entry = {
				'prediction_data' :[{
					'stock_name' : st_name,
					'start_date' : start,
					'end_date' : end,
					'prediction' : t1[0]
				}]
			}

	except Exception as e:
		print(e)


# api.add_resource(hello, '/stock', endpoint='stock')
@cron.interval_schedule(seconds=10)
def job_function():
	prediction_queue(request_queue) 

@app.route('/api/prediction', methods=['POST'])
@cross_origin()
def prediction():
	request2 = {}
	try:
		print("--> request for prediction received")
		request2['uid'] = request.json.get('uid')
		request2['st_name'] = request.json.get('st_name')
		request2['start'] = request.json.get('start')
		request2['end'] = request.json.get('end')
		global request_queue
		request_queue.append(request2)
		print("--> request queued")
		return jsonify("success"), 200

	except Exception as e:
		return f"An Error Occured: {e}"

	return jsonify(result)
	


if __name__ == '__main__':
    app.run()