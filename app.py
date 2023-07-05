from flask import Flask, request, jsonify
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pickle
import numpy as np
from datetime import datetime
from flask_cors import cross_origin
from bson import ObjectId

app = Flask(__name__)

model = pickle.load(open("./model/modelsoftmax.pkl", "rb"))

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
    data["opp_code"] = int(data["opp_code"])
    
    latest_data = collection.find({'home_code': data['home_code']}, sort=[('date', -1)]).limit(20)
    latest_data_list = list(latest_data)
    
    # Memeriksa apakah ada data pertandingan yang tersedia
    if len(latest_data_list) < 20:
        return jsonify({"Message": "Not enough data available"})
    
    # Menghitung rata-rata dari 10 pertandingan terakhir
    round_avg = sum([int(d['round']) for d in latest_data_list]) / 20
    gf_avg = sum([float(d['gf']) for d in latest_data_list]) / 20
    ga_avg = sum([float(d['ga']) for d in latest_data_list]) / 20
    xg_avg = sum([float(d['xg']) for d in latest_data_list]) / 20
    xga_avg = sum([float(d['xga']) for d in latest_data_list]) / 20
    poss_avg = sum([float(d['poss']) for d in latest_data_list]) / 20
    formation_avg = sum([float(d['formation']) for d in latest_data_list]) / 20
    sh_avg = sum([float(d['sh']) for d in latest_data_list]) / 20
    sot_avg = sum([float(d['sot']) for d in latest_data_list]) / 20
    dist_avg = sum([float(d['dist']) for d in latest_data_list]) / 20
    fk_avg = sum([float(d['fk']) for d in latest_data_list]) / 20
    pk_avg = sum([float(d['pk']) for d in latest_data_list]) / 20
    pkatt_avg = sum([float(d['pkatt']) for d in latest_data_list]) / 20
    venue_code = latest_data_list[0]['venue_code']
    
    tanggal_waktu = data["date"]
    dt = datetime.strptime(tanggal_waktu, "%Y-%m-%d")
    day = dt.day
    
    data_input = [
        round_avg, gf_avg, ga_avg, xg_avg, xga_avg, poss_avg, formation_avg, sh_avg, sot_avg,
        dist_avg, fk_avg, pk_avg, pkatt_avg, venue_code, data['opp_code'], int(data['hour']), day, int(data['home_code'])
    ]
    
    data_input = np.array(data_input).reshape(-1, 1, 18)
    data_input = np.reshape(data_input, (data_input.shape[0], -1, 1))
    
    prediction = model.predict(data_input)
    home_predict = float(prediction[0][0]) * 100
    away_predict = float(prediction[0][1]) * 100
    draw_predict = float(prediction[0][2]) * 100
    detail = f'Hasil prediksi pertandingannya yaitu home team memiliki peluang sebesar: {home_predict}%, sedangkan away team memiliki peluang sebesar: {away_predict}%, dan peluang terjadinya draw: {draw_predict}%'
    group_concat = "Group " + str(data["group"])

    if data["opp_code"] == data["home_code"]:
        return jsonify({"Message": "The Team Cannot Be the Same"})
    else:
        pred_data = {
            "date": datetime.now(),
            "home_code": int(data["home_code"]),
            "opp_code": int(data["opp_code"]),
            "home_prediction": home_predict,
            "away_prediction": away_predict,
            "draw_prediction" : draw_predict,
            "match_date": tanggal_waktu,
            "detail" : detail,
            "group" : group_concat
        }
        pred_collection.insert_one(pred_data)

    home_klasemen = klasemen_collection.find_one({"team_id": int(data["home_code"])})
    opp_klasemen = klasemen_collection.find_one({"team_id": int(data["opp_code"])})
    jumlah_data_home = pred_collection.count_documents({"home_code": int(data["home_code"])})
    jumlah_data_opp = pred_collection.count_documents({"opp_code": int(data["opp_code"])})

    if home_klasemen is not None:
        multiply_to_count_home = float(home_klasemen["total_pred"]) * (jumlah_data_home - 1)
        total_data_home = home_predict + multiply_to_count_home
        avg_home = total_data_home / jumlah_data_home
        filter_home_query = {"team_id": int(data["home_code"])}
        update_home_query = {"$set": {"total_pred": avg_home}}
        klasemen_collection.update_one(filter_home_query, update_home_query)

    if opp_klasemen is not None:
        multiply_to_count_opp = float(opp_klasemen["total_pred"]) * (jumlah_data_opp - 1)
        total_data_opp = away_predict + multiply_to_count_opp
        avg_opp = total_data_opp / jumlah_data_opp
        filter_opp_query = {"team_id": int(data["opp_code"])}
        update_opp_query = {"$set": {"total_pred": avg_opp}}
        klasemen_collection.update_one(filter_opp_query, update_opp_query)

    if home_klasemen is None:
        klasemen_data_home = {
            "team_id": int(data["home_code"]),
            "total_pred": home_predict
        }
        klasemen_collection.insert_one(klasemen_data_home)

    if opp_klasemen is None:
        klasemen_data_opp = {
            "team_id": int(data["opp_code"]),
            "total_pred": away_predict
        }
        klasemen_collection.insert_one(klasemen_data_opp)

    # Update avg_opp for the opponent team
    if home_klasemen is not None and opp_klasemen is None:
        filter_opp_query = {"team_id": int(data["opp_code"])}
        update_opp_query = {"$set": {"total_pred": avg_opp}}
        klasemen_collection.update_one(filter_opp_query, update_opp_query)

    # return jsonify({"Pesan": "Tambah baru"})

    return jsonify({"pesan" : str(avg_home) + "dan" + str(avg_opp) })
        


@app.route("/api/predictions", methods=['GET'])
@cross_origin()
def get_predictions():
    predictions = []
    for prediction in pred_collection.find():
        prediction_data = {
            "id": str(prediction["_id"]),
            "home_prediction": prediction["home_prediction"],
            "away_prediction": prediction["away_prediction"],
            "draw_prediction" : prediction["draw_prediction"],
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
            "draw_prediction" : prediction["draw_prediction"],
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
        prev_home_value = data_prediction["home_prediction"] 
        latest_home_value = check_home_id["total_pred"]
        new_home_value = (latest_home_value * jumlah_data_home - prev_home_value)/(jumlah_data_home - 1)

        prev_opp_value = data_prediction["away_prediction"]
        latest_opp_value = check_opp_id["total_pred"]
        new_opp_value = (latest_opp_value * jumlah_data_opp - prev_opp_value)/(jumlah_data_opp - 1)

    
    filter_home_query = {"team_code": home_id}
    update_home_query = {"$set": {"total_pred": new_home_value}}
    klasemen_collection.update_one(filter_home_query, update_home_query)

    filter_opp_query = {"team_code": opp_id}
    update_opp_query = {"$set": {"total_pred": new_opp_value}}
    klasemen_collection.update_one(filter_opp_query, update_opp_query)
    
    pred_collection.delete_one({"_id" : ObjectId(prediction_id)})   

@app.route("/api/klasemen", methods=['GET'])
@cross_origin()
def get_klasemen():
    klasemens = []
    for klasemen in klasemen_collection.find():
        klasemen_data = {
            "team_id": klasemen["team_id"],
            "total_pred": klasemen["total_pred"]
        }
        klasemens.append(klasemen_data)

    return jsonify(klasemens)

if __name__ == '__main__':
    app.run(debug=True)
