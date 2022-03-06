import pandas as pd
import numpy as np
import warnings
from pandas.io import gbq

def fetch_london_data(query_string, project_id, location):
    """
    This function fetches the london fire brigade data from Google BigQuery
    
    params:
        query_string - the string the contains the SQL code to query the data
        project-id - the unique project id for the Google BigQuery data
        location - where the data is being saved in Google BigQuery
        
    returns:
        data - queueried data
    """
    
    data = gbq.read_gbq(query_string, project_id = project_id, location=location)
    
    return data

def make_emergencies(df):
    """
    This function combines the two different emergencies found in the data set. It then creates a new indicator column that
    contains 1 if the observation is a real emergency and 0 if not
    
    params:
        df - dataframe with the necessary columns
    """
    emergency = ["Special Service", "Fire"]
    df["incident_group"].replace(emergency, "Emergency", inplace = True)
    df["Emergency"] = (df["incident_group"] == "Emergency").astype(np.int64)
    
    
def create_month(df):
    """
    This function creates a column in the dataframe containing the month
    
    params:
        df - dataframe with necessary columns
    """
    df["Month"] = df["timestamp_of_call"].dt.month.astype(np.str)
    
def create_hour(df):
    """
    This function creates a column in the dataframe containing the hour
    
    params:
        df - dataframe with necessary columns
    """
    df["Hour"] = df["timestamp_of_call"].dt.hour.astype(np.str)
    
def merge_property(df):
    """
    This function combines categories found in the property category variable
    
    params:
        df - dataframe with the necessary variable
        
    """
    res = ["Dwelling", "Other Residential"]
    df["property_category"].replace(res, "Residential", inplace = True)

    vehicle = ["Road Vehicle", "Aircraft", "Boat", "Rail Vehicle"]
    df["property_category"].replace(vehicle, "Vehicle", inplace = True)

    df["property_category"].replace("Outdoor Structure", "Outdoor", inplace = True)
    

def merge_address(df):
    """
    This function combines some of the categories together in the address qualifier variable.
    
    params:
        df - dataframe with the necessary columns
        
    """
    street = ["In street outside gazetteer location", "In street remote from gazetteer location", 
              "In street close to gazetteer location", "Open land/water - nearest gazetteer location"]
    df["address_qualifier"].replace(street, "Gazetter", inplace = True)

    nearby = ["Nearby address - no building in street", "Nearby address - street not listed in gazetteer"]
    df["address_qualifier"].replace(nearby, "Nearby Address", inplace = True)

    other = ["On motorway / elevated road", "Railway land or rolling stock"]
    df["address_qualifier"].replace(other, "Other", inplace = True)
    
def drop_boroughs(df):
    """
    This function drops the boroughs that have missing values
    """
    df["borough_name"].replace(" NOT GEO-CODED", np.nan, inplace=True)
    df.dropna(subset=["borough_name"], inplace = True)


def arriving_time(df):
    """
    This function removes the values that are more than 4 standard deviations above the mean and imputes the missing values
    based on the station mean
    """
    pump_mean = np.mean(df["first_time"])
    pump_std = np.std(df["first_time"])
    #Any pump times above this should be flagged
    cutoff_time = pump_mean + 4*pump_std
    
    df = df.loc[(df["first_time"] < cutoff_time) | df["first_time"].isnull()]
    
    df["first_time"] = df["first_time"].astype(np.float64)
    df["first_time"] = df.groupby("borough_name")["first_time"].\
                                                transform(lambda grp: grp.fillna(np.mean(grp)))
    return df

def station_pumps(df):
    """
    This function removes any stations that have more than 5 pumps attending and imputes the missing values with the
    mean number of pump by stations.
    
    params:
        df - dataframe with the necessary columns
    """
    df = df.loc[(df["station_pumps"] < 5) | (df["station_pumps"].isnull())]
    df["station_pumps"] = df["station_pumps"].astype(np.float64)
    df["station_pumps"] = df.groupby("borough_name")["station_pumps"].\
                                                transform(lambda grp: grp.fillna(np.mean(grp)))
    df["station_pumps"] = np.round(df["station_pumps"]).astype(np.str)
    return df
    
def pumps_attending(df):
    """
    This function removes any pumps that are greater than 5 and imputes the missing number of pumps with the mean number of 
    pumps per station
    
    params:
        df - dataframe with necessary columns
    """
    df = df.loc[(df["pumps_attending"] < 5) | (df["pumps_attending"].isnull())]
    df["pumps_attending"] = df["pumps_attending"].astype(np.float64)
    df["pumps_attending"] = df.groupby("borough_name")["pumps_attending"].\
                                                transform(lambda grp: grp.fillna(np.mean(grp)))
    df["pumps_attending"] = np.round(df["pumps_attending"]).astype(np.str).astype(np.str)
    return df
    
def clean_london(df, add_emergencies=True, make_month=True, make_hour=True, clean_property=True,
                clean_address=True, clean_boroughs=True, clean_arriving_time=True, clean_station_pumps=True, 
                 clean_pumps_attending=True, verbose=False):
    
    if verbose: print("Cleaning London Data Started...\n")
        
    if add_emergencies:
        make_emergencies(df)
        if verbose: print("Sucessfully Added Emergency Column!")
            
    if make_month:
        create_month(df)
        if verbose: print("Sucessfully Added Month Column!")
            
    if make_hour:
        create_hour(df)
        if verbose: print("Sucessfully Added Hour Column!")
            
    if clean_property:
        merge_property(df)
        if verbose: print("Sucessfully Cleanded Property Category!")
            
    if clean_address:
        merge_address(df)
        if verbose: print("Sucessfully Cleaned Address!")
            
    if clean_boroughs:
        drop_boroughs(df)
        if verbose: print("Sucessfully Cleaned Borough Names!")
            
    if clean_arriving_time:
        df = arriving_time(df)
        if verbose: print("Sucessfully Cleaned Station Arriving Time!")
            
    if clean_station_pumps:
        df = station_pumps(df)
        if verbose: print("Sucessfully Cleaned Number of Stations with Pumps!")
            
    if clean_pumps_attending:
        df = pumps_attending(df)
        if verbose: print("Sucessfully Cleaned Number of Pumps Attending!")
            
    df = df.drop(["incident_number", "incident_group", "timestamp_of_call"], axis = 1, inplace = False)
    if verbose: print("Sucessfully Dropped Unecessary Columns!")
    
    if verbose: print("\nSucessfully Cleaned London Data!")
        
    return df
    
        
    
