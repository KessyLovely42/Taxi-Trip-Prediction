#serialization
import pickle

#data preprocessing
from sklearn.linear_model import LinearRegression
from data_processing import *


def train_model(X_train, Y_train):

    lr = LinearRegression()
    lr.fit(X_train, Y_train)

    return lr

# main function containing training pipeline

def main():
    train_data = data_ingestion()
    train_data = create_target_label(train_data)
    train_data = treat_na(train_data)
    train_data = treat_outliers(train_data, year=2023, month_number=1)
    train_data = engineer_features(train_data)
    X, Y = preprocess_data(train_data)

    #train model
    final_model = train_model(X,Y)

    #serialize model
    file_name = "model_store/final_model.pkl"
    with open(file_name,"wb") as outfile:
        pickle.dump(final_model,outfile)

    model_path = "/Users/okundiakessy/Documents/All-things-Python/taxi-trip-duration-prediction/Taxi-Trip-Duration-Prediction-App/taxi_trip_duration_prediction_app/model_store/final_model.pkl"
    with open(model_path,"rb") as model_file:
    model = pickle.load(model_file)
    y = model.predict(X)
#main function
if __name__ == "__main__":
    main()

