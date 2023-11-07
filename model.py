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
    df = data.copy()
    df_clean = data.copy()
    #-----------------------------------------------------------------------
    # Dataframe for each feature
    #-----------------------------------------------------------------------
    wind_speed_columns = [col for col in df.columns if col.endswith('_wind_speed')]
    df_wind_speed = df[wind_speed_columns]
    wind_deg_columns = [col for col in df.columns if col.endswith('_wind_deg')]
    df_wind_deg = df[wind_deg_columns]
    rain_1h_columns = [col for col in df.columns if col.endswith('_rain_1h')]
    df_rain_1h = df[rain_1h_columns]
    rain_3h_columns = [col for col in df.columns if col.endswith('_rain_3h')]
    df_rain_3h = df[rain_3h_columns]
    humidity_columns = [col for col in df.columns if col.endswith('_humidity')]
    df_humidity = df[humidity_columns]
    clouds_all_columns = [col for col in df.columns if col.endswith('_clouds_all')]
    df_clouds_all = df[clouds_all_columns]
    pressure_columns = [col for col in df.columns if col.endswith('_pressure')]
    df_pressure = df[pressure_columns]
    snow_3h_columns = [col for col in df.columns if col.endswith('_snow_3h')]
    df_snow_3h = df[snow_3h_columns]
    temp_columns = [col for col in df.columns if col.endswith('_temp')]
    df_temp = df[temp_columns]
    temp_max_columns = [col for col in df.columns if col.endswith('_temp_max')]
    df_temp_max = df[temp_max_columns]
    temp_min_columns = [col for col in df.columns if col.endswith('_temp_min')]
    df_temp_min = df[temp_min_columns]
    weather_id_columns = [col for col in df.columns if col.endswith('_weather_id')]
    df_weather_id = df[weather_id_columns]
    #-----------------------------------------------------------------------


    #-----------------------------------------------------------------------
    # Dataframe for each city
    #-----------------------------------------------------------------------
    madrid_columns = [col for col in df.columns if col.startswith('Madrid')]
    madrid_columns.append('load_shortfall_3h')
    df_madrid = df[madrid_columns]
    barcelona_columns = [col for col in df.columns if col.startswith('Barcelona')]
    barcelona_columns.append('load_shortfall_3h')
    df_barcelona = df[barcelona_columns]
    valencia_columns = [col for col in df.columns if col.startswith('Valencia')]
    valencia_columns.append('load_shortfall_3h')
    df_valencia = df[valencia_columns]
    seville_columns = [col for col in df.columns if col.startswith('Seville')]
    seville_columns.append('load_shortfall_3h')
    df_seville = df[seville_columns]
    bilbao_columns = [col for col in df.columns if col.startswith('Bilbao')]
    bilbao_columns.append('load_shortfall_3h')
    df_bilbao = df[bilbao_columns]  
    #-------------------------------------------------------------------------
    grouped_means = df_clean.groupby(['Seville_pressure'])['Valencia_pressure'].transform('mean')
    default_fill_value = df['Valencia_pressure'].mean()
    df_clean['Valencia_pressure'] = np.where(grouped_means.isna(), default_fill_value, grouped_means)

    df_clean['time'] = pd.to_datetime(df_clean['time'])  #Change

    wind_deg_map = {
    'level_1': 1,
    'level_2': 2,
    'level_3': 3,
    'level_4': 4,
    'level_5': 5,
    'level_6': 6,
    'level_7': 7,
    'level_8': 8,
    'level_9': 9,
    'level_10': 10
    }
    df_wind_deg.Valencia_wind_deg = df.Valencia_wind_deg.replace(wind_deg_map)
    df_clean.Valencia_wind_deg = df_wind_deg.Valencia_wind_deg * 40.0
    df_clean.Valencia_wind_deg = df_clean.Valencia_wind_deg.sub(40)

    pressure_map = {
    'sp1': 1,
    'sp2': 2,
    'sp3': 3,
    'sp4': 4,
    'sp5': 5,
    'sp6': 6,
    'sp7': 7,
    'sp8': 8,
    'sp9': 9,
    'sp10': 10,
    'sp11': 11,
    'sp12': 12,
    'sp13': 13,
    'sp14': 14,
    'sp15': 15,
    'sp16': 16,
    'sp17': 17,
    'sp18': 18,
    'sp19': 19,
    'sp20': 20,
    'sp21': 21,
    'sp22': 22,
    'sp23': 23,
    'sp24': 24,
    'sp25': 25
    }
    df_pressure.Seville_pressure = df.Seville_pressure.replace(pressure_map)
    df_clean.Seville_pressure = df_pressure.Seville_pressure * 2.888889
    df_clean.Seville_pressure = df_clean.Seville_pressure.add(969.78)

    pd.concat([df_clean.Seville_pressure,df_pressure.Seville_pressure, df.Seville_pressure], axis=1)

    date_strings = df.time
    date_objects = date_strings.apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))

    df_clean['day'] = date_objects.apply(lambda x: x.day)
    df_clean['month'] = date_objects.apply(lambda x: x.month)
    df_clean['year'] = date_objects.apply(lambda x: x.year)

    df_clean.drop('time', axis=1)
    
    # features dataset
    X = df_clean.drop(['time','load_shortfall_3h'], axis=1)

    # target dataset
    y = df_clean['load_shortfall_3h']
    # ------------------------------------------------------------------------
    predict_vector = df_clean.copy()

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
