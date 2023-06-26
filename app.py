from flask import Flask, request, jsonify
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pickle
import numpy as np
from datetime import datetime
from flask_cors import cross_origin
from bson import ObjectId

app = Flask(__name__)

model = pickle.load(open("./model/model3.pkl", "rb"))

uri = "mongodb://localhost:27017"
client = MongoClient(uri, server_api=ServerApi('1'))
database = client['football-predict']
collection = database['master']
pred_collection = database['prediksi']
klasemen_collection = database['klasemen']

@app.route("/api/predict", methods=['POST'])
@cross_origin()
def predict():
    data = request.get_json()
    data["home_code"] = int(data["home_code"])
    
    latest_data = collection.find_one({'home_code': data['home_code']}, sort=[('date', -1)])
    tanggal_waktu = data["date"]
    dt = datetime.strptime(tanggal_waktu, "%Y-%m-%d")
    day = dt.day
    
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
    data_input = data_input.reshape((1, 1, 27))
    
    prediction = model.predict(data_input)
    prediction = float(prediction)
    away_prediction = 100 - prediction
    detail = f'Hasil prediksi pertandingannya yaitu home team memiliki peluang sebesar : {prediction}, sedangkan away team memiliki peluang sebesar :   {away_prediction}'
    group_concat = "Group " + str(data["group"])


    if data["opp_code"] == data["home_code"]:
        return jsonify({"Message": "The Team Cannot Be the Same"})
    else:
        pred_data = {
            "date": datetime.now(),
            "home_code": int(data["home_code"]),
            "opp_code": int(data["opp_code"]),
            "home_prediction": prediction,
            "away_prediction": away_prediction,
            "match_date": tanggal_waktu,
            "detail" : detail,
            "group" : group_concat
        }
        pred_collection.insert_one(pred_data)

        return jsonify({"Home Prediction": prediction, "Away Prediction": away_prediction})
    
@app.route("/api/predictions", methods=['GET'])
@cross_origin()
def get_predictions():
    predictions = []
    for prediction in pred_collection.find():
        prediction_data = {
            "id": str(prediction["_id"]),
            "home_prediction": prediction["home_prediction"],
            "away_prediction": prediction["away_prediction"],
            "group": prediction["group"],
            "match_date": prediction["match_date"],
            "home_code": prediction["home_code"],
            "opp_code": prediction["opp_code"],
            "detail": prediction["detail"]
        }
        predictions.append(prediction_data)

    return jsonify(predictions)

@app.route("/api/predictions/<prediction_id>", methods=['GET'])
@cross_origin()
def get_prediction(prediction_id):
    prediction = pred_collection.find_one({"_id": ObjectId(prediction_id)})
    if prediction:
        prediction_data = {
            "id": str(prediction["_id"]),
            "home_prediction": prediction["home_prediction"],
            "away_prediction": prediction["away_prediction"],
            "group": prediction["group"],
            "match_date": prediction["match_date"],
            "home_code": prediction["home_code"],
            "opp_code": prediction["opp_code"],
            "detail": prediction["detail"]
        }
        return jsonify(prediction_data)
    else:
        return jsonify({"Message": "Prediction not found"})

@app.route('/api/delete/<prediction_id>', methods=['DELETE'])
@cross_origin()
def delete_data(prediction_id):
    data_prediction = pred_collection.find_one({"_id" : ObjectId(prediction_id)})
    home_id = data_prediction["home_code"]
    opp_id = data_prediction["opp_code"]
    check_home_id = klasemen_collection.find_one({"team_id" : home_id})
    check_opp_id = klasemen_collection.find_one({"team_id" : opp_id})

    jumlah_data_home = pred_collection.count_documents({"home_code" : home_id})
    jumlah_data_opp = pred_collection.count_documents({"opp_code" : opp_id})

    if check_home_id  and check_opp_id:
        prev_home_value = data_prediction["home_prediction"] * 100
        latest_home_value = check_home_id["total_pred"] * 100
        new_home_value = (latest_home_value * jumlah_data_home - prev_home_value)/(jumlah_data_home - 1)
        id_filter = home_id

        prev_opp_value = data_prediction["away_prediction"] * 100
        latest_opp_value = check_opp_id["total_pred"] * 100
        new_opp_value = (latest_opp_value * jumlah_data_opp - prev_opp_value)/(jumlah_data_opp - 1)

    
    filter_home_query = {"team_code": home_id}
    update_home_query = {"$set": {"total_pred": new_home_value}}
    klasemen_collection.update_one(filter_home_query, update_home_query)

    filter_opp_query = {"team_code": opp_id}
    update_opp_query = {"$set": {"total_pred": new_opp_value}}
    klasemen_collection.update_one(filter_opp_query, update_opp_query)
    
    pred_collection.delete_one({"_id" : ObjectId(prediction_id)})

if __name__ == '__main__':
    app.run(debug=True)
