# Creating an API which simply returns the ASCII value of the character passed from the flutter app.
from flask import Flask, request, jsonify
import pickle
import pandas as pd

ipl_matches = pd.read_csv("data/ipl_matches.csv")
with open("win_prediction_ipl.pkl", "rb") as f:
    win_prediction_ipl = pickle.load(f)

with open("score_prediction_ipl.pkl", 'rb') as f:
    scorepred = pickle.load(f)

batsman_info = pd.read_csv("data/cricket_batsman_information.csv")
with open("batsman_prediction.pkl", "rb") as f:
    batsman_prediction = pickle.load(f)

winpred = win_prediction_ipl["estimator"]
encoders = win_prediction_ipl["encoders"]

bat_pred = batsman_prediction["estimator"]
encoders_x = batsman_prediction["encoders"]
app = Flask(__name__)

@app.route('/ipl/win', methods = ['GET'])
def winpredictionipl():
    args = dict(request.args)
    print(args)
    city = encoders['city'].transform([args.get("city")])[0]
    team1 = encoders['team1'].transform([args.get("team1")])[0]
    team2 = encoders['team2'].transform([args.get("team2")])[0]
    toss_winner = encoders['toss_winner'].transform([args.get("toss_winner")])[0]
    toss_decision = encoders['toss_decision'].transform([args.get("toss_decision")])[0]

    X = [city, team1, team2, toss_winner, toss_decision]
    print(X)
    print(winpred.predict([X]))
    return list(encoders["winner"].inverse_transform([winpred.predict([X])]))

@app.route('/ipl/score', methods=["GET"])
def scoreprediction():
    runs = request.args.get("runs")
    wickets = request.args.get("wickets")
    overs = request.args.get("overs")
    runs_last_5 = request.args.get("runs_last_5")
    striker = request.args.get("striker")
    non_striker = request.args.get("non_striker")


    X = [runs, wickets, overs, runs_last_5, striker, non_striker]
    print(X)
    return list(scorepred.predict([X]))

@app.route('/player/performance', methods = ['GET'])
def playerprediction():
    args = dict(request.args)
    print(args)
    player_name = encoders_x['Innings Player'].transform([args.get("player_name")])[0]
    opposition_name = encoders_x['Opposition_x'].transform([args.get("opposition_name")])[0]
    ground = encoders_x['Ground'].transform([args.get("ground")])[0]
    country = encoders_x['Country_x'].transform([args.get("country")])[0]
    no_of_innings = float(request.args.get("no_of_innings"))
    average = float(request.args.get("average"))
    strike_rate = float(request.args.get("strike_rate"))
    highest_score = float(request.args.get("highest_score"))
    zeros = float(request.args.get("zeros"))
    fifties = float(request.args.get("fifties"))
    centuries = float(request.args.get("centuries"))

    Z = [player_name, opposition_name, ground, country, strike_rate, average, fifties, centuries, zeros, no_of_innings]
    print(Z)

    consistency_x = 0.4262*(average) + 0.2566*(no_of_innings) + 0.1510*(strike_rate) + 0.0787*(centuries) + 0.0556*(fifties) - 0.0328*(zeros)
    form_x = 0.4262*(average) + 0.2566*(no_of_innings) + 0.1510*(strike_rate) + 0.0787*(centuries) + 0.0556*(fifties) - 0.0328*(zeros)
    opposition_x = 0.4262*(average) + 0.2566*(no_of_innings) + 0.1510*(strike_rate) + 0.0787*(centuries) + 0.0556*(fifties) - 0.0328*(zeros)
    venue_x = 0.4262*(average) + 0.2566*(no_of_innings) + 0.1510*(strike_rate) + 0.0787*(centuries) + 0.0556*(fifties) + 0.0328*(highest_score)



    if(consistency_x >= 1 and consistency_x <= 49):
        consistency = 1
    elif(consistency_x >= 50 and consistency_x <= 99):
        consistency = 2
    elif(consistency_x >= 100 and consistency_x <= 124):
        consistency = 3
    elif(consistency_x >= 124 and consistency_x <= 149):
        consistency = 4
    elif(consistency_x >= 150):
        consistency = 5
    
    if(form_x >= 1 and form_x <= 4):
        form = 1
    elif(form_x >= 5 and form_x <= 9):
        form = 2
    elif(form_x >= 10 and form_x <= 11):
        form = 3
    elif(form_x >= 12 and form_x <= 14):
        form = 4
    elif(form_x >= 15.5):
        form = 5
    
    if(opposition_x >= 1 and opposition_x <= 2):
        opposition = 1

    elif(opposition_x >= 3 and opposition_x <= 4):
        opposition = 2
    elif(opposition_x >= 5 and opposition_x <= 6):
        opposition = 3
    elif(opposition_x >= 7 and opposition_x <= 9):
        opposition = 4
    elif(opposition_x >= 10):
        opposition = 5
    
    if(venue_x == 1):
        venue = 1
    elif(venue_x == 2):
        venue = 2
    elif(venue_x == 3):
        form = 3
    elif(venue_x == 4):
        venue = 4
    elif(venue_x >= 5):
        venue = 5


    X = [player_name, opposition_name, ground, country, strike_rate, average, consistency, form, opposition, venue, fifties, centuries, zeros, no_of_innings]
    print(X)
    y = bat_pred.predict([X])
    
    if(y == 1.0):
        return ["He will score runs in range of 1 - 24"]
    elif(y == 2.0):
        return ["He will score runs in range of 25 - 49"]
    elif(y == 3.0):
        return[ "He will score runs in range of 50-74"]
    elif(y == 4.0):
        return ["He will score runs in range of 75 - 99"]
    elif(y == 5.0):
        return ["He will score runs more than 100"]

if __name__ == "__main__":
    app.run(debug=True)
