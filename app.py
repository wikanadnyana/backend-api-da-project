from flask import Flask, request, jsonify
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pickle
import numpy as np
from datetime import datetime
from flask_cors import cross_origin


app = Flask(__name__)

model = pickle.load(open("./model/model.pkl", "rb"))

uri = "mongodb://localhost:27017"



client = MongoClient(uri, server_api=ServerApi('1'))
database = client['football-predict']
collection = database['master']

@app.route("/predict", methods=['POST'])
@cross_origin()
def predict():
    data = request.get_json()
    data["home_code"] = int(data["home_code"])
    # data.headers.add("Access-Control-Allow-Origin", "*")
    latest_data = collection.find_one({'home_code': data['home_code']}, sort=[('date', -1)])
    tanggal_waktu = data["date"]
    dt = datetime.strptime(tanggal_waktu, "%Y-%m-%d")
    day = dt.day
    # if latest_data is not None:
    data_input = [
            int(latest_data['round']), float(latest_data['venue']), float(latest_data['gf']), float(latest_data['ga']),
            float(latest_data['xg']), float(latest_data['xga']), float(latest_data['poss']),
            float(latest_data['formation']), float(latest_data['sh']), float(latest_data['sot']),
            float(latest_data['dist']), float(latest_data['fk']), float(latest_data['pk']), float(latest_data['pkatt']),
            float(latest_data['venue_code']), int(data['opp_code']), int(data['hour']), day,
            float(latest_data['gf_rolling']), float(latest_data['ga_rolling']), float(latest_data['sh_rolling']),
            float(latest_data['sot_rolling']), float(latest_data['dist_rolling']), float(latest_data['fk_rolling']),
            float(latest_data['pk_rolling']), float(latest_data['pkatt_rolling']), int(data['home_code'])
        ]
    data_input = np.array(data_input)
    data_input = data_input.reshape(1, -1)
    data_input= data_input.reshape((1, 1, 27))
    prediction = model.predict(data_input)
    prediction = float(prediction)
    away_prediction = 100 - prediction
    # else :
        # return jsonify({"Message": type(data["home_code"])})

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