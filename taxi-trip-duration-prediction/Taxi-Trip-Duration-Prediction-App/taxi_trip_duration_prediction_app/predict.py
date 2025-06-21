
import json
from data_processing import *
import pandas as pd
import numpy as np

#load json file containing user input
json_filepath = "/Users/okundiakessy/Documents/All-things-Python/taxi-trip-duration-prediction/Taxi-Trip-Duration-Prediction-App/taxi_trip_duration_prediction_app/inference_store/user_input.json"
with open(json_filepath,"rb") as input_file:
    input = json.load(input_file)

#load scaler object
scaler_obj_path= "/Users/okundiakessy/Documents/All-things-Python/taxi-trip-duration-prediction/Taxi-Trip-Duration-Prediction-App/taxi_trip_duration_prediction_app/model_store/transformation_obj.pkl"
with open(scaler_obj_path,"rb") as scaler_file:
    scaler = pickle.load(scaler_file)

#load model object
model_path = "/Users/okundiakessy/Documents/All-things-Python/taxi-trip-duration-prediction/Taxi-Trip-Duration-Prediction-App/taxi_trip_duration_prediction_app/model_store/final_model.pkl"
with open(model_path,"rb") as model_file:
    model = pickle.load(model_file)


def predict(input):
    df_1 = pd.DataFrame(input, index=[0])

    df_1["tpep_pickup_datetime"] = pd.to_datetime(df_1["tpep_pickup_datetime"])
    #Get day of week and hour of the day from trip pick up time
    df_1["day_of_week"] = df_1["tpep_pickup_datetime"].dt.day_of_week
    df_1["hour_of_day"] = df_1["tpep_pickup_datetime"].dt.hour

    #drop the pickup and drop off time from the dataframe as the day and hour have been extracted
    df_1 = df_1.drop(columns=["tpep_pickup_datetime"])

    #encode the store_and_fwd_flag
    df_1["store_and_fwd_flag"] = df_1["store_and_fwd_flag"].replace(to_replace=["Y","N"], value=[1,2])

    #scale input
    X = scaler.transform(df_1)

    #predict
    y = model.predict(X)
    print(y)

    
def main():
    predict(input)

if __name__ == "__main__":
    main()