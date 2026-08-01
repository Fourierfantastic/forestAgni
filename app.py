from flask import Flask,request,render_template
import requests
from flask import jsonify,request 
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
        day=int(request.form.get("day")),
        month=int(request.form.get("month")),
        year=int(request.form.get("year")),
        Temperature=float(request.form.get("Temperature")),
        RH=float(request.form.get("RH")),
        Ws=float(request.form.get("Ws")),
        Rain=float(request.form.get("Rain")),
        FFMC=float(request.form.get("FFMC")),
        DMC=float(request.form.get("DMC")),
        DC=float(request.form.get("DC")),
        ISI=float(request.form.get("ISI")),
        BUI=float(request.form.get("BUI")),
        Region=int(request.form.get("Region")),
        Classes=request.form.get("Classes")
    )
    
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        fwi = round(float(results[0]),2)

        if fwi < 5:
            danger = "🟢 Very Low"
            image = "very_low.jpg"
        elif fwi < 15:
            danger = "🟡 Low to Moderate"
            image = "low.jpg"
        elif fwi < 30:
            danger = "🟠 High"
            image = "high.jpg"
        elif fwi < 45:
            danger = "🔴 Very High"
            image = "very_high.jpg"
        else:
            danger = "🚨 Extreme"
            image = "extreme.jpg"

        return render_template(
        "home.html",
        results=fwi,
        danger=danger,
        image = image
    )

 # current data weather
@app.route("/current_weather")
def current_weather():

    city = request.args.get("city")

    if not city:
        return jsonify({"error": "Please enter a city name."})

    # -------- Geocoding API --------

    geo_url = (
        f"https://geocoding-api.open-meteo.com/v1/search"
        f"?name={city}&count=1"
    )

    geo_response = requests.get(geo_url)
    geo_data = geo_response.json()

    if "results" not in geo_data:
        return jsonify({"error": "City not found."})

    latitude = geo_data["results"][0]["latitude"]
    longitude = geo_data["results"][0]["longitude"]
    location = geo_data["results"][0]["name"]
    country = geo_data["results"][0]["country"]

    # -------- Weather API --------

    weather_url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitude}"
        f"&longitude={longitude}"
        f"&current=temperature_2m,"
        f"relative_humidity_2m,"
        f"wind_speed_10m,"
        f"rain"
    )

    weather_response = requests.get(weather_url)
    weather_data = weather_response.json()

    current = weather_data["current"]

    return jsonify({

        "location": location,
        "country": country,

        "temperature": current["temperature_2m"],

        "humidity": current["relative_humidity_2m"],

        "wind_speed": current["wind_speed_10m"],

        "rain": current["rain"]

    })
if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)        


