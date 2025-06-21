#import required libraries
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def data_ingestion():
    yellow_taxi_trip_jan_2023_path = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"
    jan_2023 = pd.read_parquet(yellow_taxi_trip_jan_2023_path)
    return jan_2023

#function to ingest data and output a pandas DataFrame
#with a target label
def create_target_label(df):
    """
    Create a target variable column by subtracting the trip pick up from drop off and converts it minutes with data type of float for easy visualization and analysis.
    """
    #create the duration 
    df["trip_duration_mins"] = df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    df["trip_duration_mins"] = df["trip_duration_mins"].dt.total_seconds().map(lambda x:round(x/60,2))

    return df

# treat missing values
def treat_na(df):
    """
    This function converts features to appropriate data types,
    checks for percentage of missing values, drops features with missing values above 49 percent
    or else fill missing values with linear interpolation for continous data, and mode for categorical data
    """ 
    cat_1 = pd.CategoricalDtype(categories=[1,2,3,4,5,6], ordered=True)
    df["RatecodeID"] = df["RatecodeID"].astype(cat_1)
    df["payment_type"] = df["payment_type"].astype(cat_1)
    df["store_and_fwd_flag"] = df["store_and_fwd_flag"].astype("category")
    df["PULocationID"] = df["PULocationID"].astype("category")
    df["DOLocationID"] = df["DOLocationID"].astype("category")

    percent = round(df.isna().sum() * 100/df.shape[0] ,2)

    #loop through item
    for index, item in percent.items():
        if item > 49:
            df = df.drop(columns=[index])
        else:
            #fill missing categorical observations with mode
            if df[index].dtype == "category":
                df[index] = df[index].fillna(df[index].mode()[0])
            else: 
                #use linear interpolation for other data types
                df[index] = df[index].fillna(df[index].median())
    return df

# function to treat outliers
def treat_outliers(df, year = None, month_number = None):
    """ 
    This functions takes in a dataframe, a year, number of month parameter, lower and upper bound percentiles. 
    Drop every observation that is outside the range of the year and month given, drops indices with negative value observations
    and finally drop values outside specified percentile bounds

    """
    num_features = df.select_dtypes(include = ["int", "float"]).columns

    # drop observations outside the date range of 2023-01-01 to 2023-01-31 
    df_1 = df.drop(index=np.where(df["tpep_pickup_datetime"].dt.year != year)[0]) 
    df_1 = df.drop(index=np.where(df["tpep_pickup_datetime"].dt.month != month_number)[0])

    #cdrop observations with negative values
    df_1= df.drop(index=np.where(df[num_features].lt(0))[0])

    #drop outliers below lower bound percentile percentile and above upper bound percentile percentile  
    outlier_indices = []
    for col in df_1.select_dtypes(include=["int64","float64"]).columns:

        lower_bound = np.percentile(df_1[col],10)
        upper_bound = np.percentile(df_1[col],95)
        outliers = np.where(df_1[col].lt(lower_bound) | df_1[col].gt(upper_bound))[0] #get indices of outliers

        outlier_indices.extend(outliers) #Append to the list outlier indices
    
    #Ensure the outliers are unique by subtracting the intersecting index elements in both dataframe indices and outlier list
    #from the set of outlier indices 
    unique_outlier_indices = set(outlier_indices) - (set(outlier_indices)-set(df_1.index.tolist()))
    
    #drop outliers
    df_1 = df_1.drop(index=unique_outlier_indices)

    return df_1

def engineer_features(df):
    """ 
    This funtion takes in a dataframe, extracts the day of the week and the hour from the pickup time 
    -drops the pickup and drop off time from the dataframe
    -encodes the the store and fwd flag feature
    
    """

    #copy dataframe into new data frame for feature engineering
    df_1 = df.copy()
    #Get day of week and hour of the day from trip pick up time
    df_1["day_of_week"] = df_1["tpep_pickup_datetime"].dt.day_of_week
    df_1["hour_of_day"] = df_1["tpep_pickup_datetime"].dt.hour

    #drop the pickup and drop off time from the dataframe as the day and hour have been extracted
    df_1 = df_1.drop(columns=["tpep_pickup_datetime","tpep_dropoff_datetime","VendorID"])

    #encode the store_and_fwd_flag
    df_1["store_and_fwd_flag"] = df_1["store_and_fwd_flag"].rename_categories([1,2])
    return df_1

def preprocess_data(df):

    rs = RobustScaler()

    df_1 = df.copy()
    #split dataframe into target and training set
    X = df_1.copy()
    Y = X.pop("trip_duration_mins")


    #Scale Features using RobustScaler
    rs.fit(X)
    X_train_transformed = rs.transform(X)

    #serialize transformation object
    file_name = "model_store/transformation_obj.pkl"
    with open(file_name,"wb") as outfile:
        pickle.dump(rs,outfile)

    return X_train_transformed,Y