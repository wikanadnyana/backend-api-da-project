from flask import Flask, request, jsonify
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pickle
import numpy as np
from datetime import datetime

app = Flask(__name__)

model = pickle.load(open("./model/model.pkl", "rb"))

uri = "mongodb://localhost:27017"



client = MongoClient(uri, server_api=ServerApi('1'))
database = client['football-predict']
collection = database['master']

@app.route("/predict", methods=['POST'])
def predict():
    data = request.get_json()
    latest_data = collection.find_one({'home_code': data['home_code']}, sort=[('date', -1)])
    tanggal_waktu = data["date"]
    dt = datetime.strptime(tanggal_waktu, "%Y-%m-%d")
    day = dt.day

    data_input = [latest_data['round'], latest_data['venue'], latest_data['gf'] , latest_data['ga'], latest_data['xg'], latest_data['xga'], latest_data['poss'],
                   latest_data['formation'], latest_data['sh'], latest_data['sot'], latest_data['dist'], latest_data['fk'], latest_data['pk'], latest_data['pkatt']
                   , latest_data['venue_code'], data['opp_code'], data['hour'], day, 
                   latest_data['gf_rolling'], latest_data['ga_rolling'],
                   latest_data['sh_rolling'], latest_data['sot_rolling'], latest_data['dist_rolling'], latest_data['fk_rolling'],
                    latest_data['pk_rolling'], latest_data['pkatt_rolling'], data['home_code']]
    data_input = np.array(data_input)
    data_input = data_input.reshape(1, -1)
    data_input= data_input.reshape((1, 1, 27))
    prediction = model.predict(data_input)
    prediction = float(prediction)
    away_prediction = 100 - prediction

    if data["opp_code"] == data["home_code"] :
        return jsonify({"Message" : "The Team Cannot Same"})
    else :
        return jsonify({"Home Prediction": prediction, "Away Prediction":away_prediction})

    # if latest_game is not None:
    #     date = latest_game['date']
    #     xg = latest_game['xg']
    #     venue = latest_game['venue']
    #     poss = latest_game['poss']
    #     print(f'date: {date}')
    #     print(f'xg: {xg}')
    #     print(f'venue: {venue}')
    #     print(f'opponent: {poss}')
    # else:
    #     print('Tidak ada data tersedia.')


if __name__ == '__main__':
    app.run(debug=True)