"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    #predict_vector = feature_vector_df[['Madrid_wind_speed','Bilbao_rain_1h','Valencia_wind_speed']]
    df_clean = data.copy()

    # Replace Barcelona pressure values outside the specified range with NaN
    df_clean['Barcelona_pressure'] = np.where(
    (df_clean['Barcelona_pressure'] > 1051) | (df_clean['Barcelona_pressure'] < 931),
    np.nan,
    df_clean['Barcelona_pressure']
    )

    # fill all null values in the Valencia pressure feature with the mode
    df_clean["Valencia_pressure"] = df_clean["Valencia_pressure"].fillna(df_clean["Valencia_pressure"].mode()[0])

    # Fill null values in the Barcelona_pressure using linear interpolation
    df_clean['Barcelona_pressure'].interpolate(method='linear', limit_direction='forward', inplace=True, axis=0)

    # changing data to correct format
    df_clean['time'] = pd.to_datetime(df_clean['time'])

    # Extracting numeric values from the "Valencia_wind_deg" column and converting to numeric data type
    df_clean["Valencia_wind_deg"] = df_clean["Valencia_wind_deg"].str.extract("(\d+)")
    df_clean["Valencia_wind_deg"] = pd.to_numeric(df_clean["Valencia_wind_deg"])

    # Extracting numeric values from the "Seville_pressure" column and converting to numeric data type
    df_clean["Seville_pressure"] = df["Seville_pressure"].str.extract("(\d+)")
    df_clean["Seville_pressure"] = pd.to_numeric(df_clean["Seville_pressure"])

    # Dropping columns in df_clean that match the regex pattern '_rain_3h$' which is the amount of rainfall in the past 3 hours
    df_clean = df_clean.drop(df_clean.filter(regex='_rain_3h$', axis=1).columns, axis=1)

    # Creating new columns for combined wind in different cities
    df_clean['barcelona_combined_wind'] = df_clean['Barcelona_wind_speed'] * df_clean['Barcelona_wind_deg']
    df_clean['bilbao_combined_wind'] = df_clean['Bilbao_wind_speed'] * df_clean['Bilbao_wind_deg']
    df_clean['valencia_combined_wind'] = df_clean['Valencia_wind_speed'] * df_clean['Valencia_wind_deg']

    # Creating new columns for cumulative rain and snow based on regex patterns
    df_clean['cumulative_rain'] = df_clean.filter(regex='_rain_1h$').sum(axis=1)
    df_clean['cumulative_snow'] = df_clean.filter(regex='_snow_3h$').sum(axis=1)

    # Creating new columns for temperature ranges in different cities
    df_clean['madrid_temperature_range'] = df_clean['Madrid_temp_max'] - df_clean['Madrid_temp_min']
    df_clean['barcelona_temperature_range'] = df_clean['Barcelona_temp_max'] - df_clean['Barcelona_temp_min']
    df_clean['valencia_temperature_range'] = df_clean['Valencia_temp_max'] - df_clean['Valencia_temp_min']
    df_clean['seville_temperature_range'] = df_clean['Seville_temp_max'] - df_clean['Seville_temp_min']
    df_clean['bilbao_temperature_range'] = df_clean['Bilbao_temp_max'] - df_clean['Bilbao_temp_min']

    # Creating new columns for wind energy calculations in different cities
    df_clean['madrid_wind_energy'] = 0.5 * 1.225 * df_clean['Madrid_wind_speed'] ** 3
    df_clean['barcelona_wind_energy'] = 0.5 * 1.225 * df_clean['Barcelona_wind_speed'] ** 3
    df_clean['valencia_wind_energy'] = 0.5 * 1.225 * df_clean['Valencia_wind_speed'] ** 3
    df_clean['seville_wind_energy'] = 0.5 * 1.225 * df_clean['Seville_wind_speed'] ** 3
    df_clean['bilbao_wind_energy'] = 0.5 * 1.225 * df_clean['Bilbao_wind_speed'] ** 3

    # Extracting date strings from the 'time' column of the DataFrame
    date_strings = df.time

    # Converting date strings to datetime objects using the specified format
    date_objects = date_strings.apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))

    # Adding a new 'season' column based on the month of each date in 'date_objects'
    df_clean['season'] = date_objects.apply(lambda x: 1 if x.month <= 3 else
                                            2 if x.month <= 6 else
                                            3 if x.month <= 9 else 4)

    # Adding new columns to df_clean based on date information in date_objects

    # Extracting the hour of the day
    df_clean['hour'] = date_objects.apply(lambda x: x.hour)

    # Extracting the day of the month
    df_clean['day'] = date_objects.apply(lambda x: x.day)

    # Extracting the weekday (Monday is 0, Sunday is 6)
    df_clean['weekday'] = date_objects.apply(lambda x: x.weekday())

    # Extracting the day of the year
    df_clean['day_of_year'] = date_objects.apply(lambda x: x.dayofyear)

    # Extracting the month
    df_clean['month'] = date_objects.apply(lambda x: x.month)

    # Extracting the year
    df_clean['year'] = date_objects.apply(lambda x: x.year)

    # Converting 'time' column to datetime format with UTC and setting it as the index
    df_clean['time'] = pd.to_datetime(df_clean['time'], utc=True, infer_datetime_format=True)
    df_clean = df_clean.set_index('time')

    
    predict_vector = df_clean.copy()
    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
