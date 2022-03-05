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
    
    
def property_type_rank(df):
    """
    This function creates a new column in the dataframe based on how the property types rank according to their percentage
    of emergencies. 
    
    params:
        df - dataframe with the necessary variables
    """
    #Group by property type and summarize by the Emergency variable
    pt_grp = df.groupby("property_type")["Emergency"].aggregate(["mean", "sum"]).sort_values(by="mean", ascending=False)
    pt_grp.reset_index(inplace = True)
    #Change the proportion into a percentage
    pt_grp["mean"] *= 100
    #Create a new variable that ranks the values based on their percentage
    pt_grp["pt_rank"] = (pt_grp["mean"] / 10).astype(np.int)
    #Specifically change the values with 0 percentage to -1 as a special category i.e no chance it's a real emergency
    pt_grp.loc[pt_grp["mean"] == 0, "pt_rank"] = -1
    
    #Join the london data with the pt_df to match the property type rankings
    df = pd.merge(left = df, right = pt_grp[["property_type", "pt_rank"]], left_on = "property_type", right_on="property_type")
    df["pt_rank"] = df["pt_rank"].astype(np.str)
    
    return df

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
    
def ward_name_rank(df):
    """
    This function ranks the wards based on their percentage of emergencies. 
    
    params:
        df - dataframe with the necessary variables
    """
    #Group by property type and summarize by the Emergency variable
    wn_grp = df.groupby("ward_name")["Emergency"].aggregate(["mean", "sum"]).sort_values(by="mean", ascending=False)
    wn_grp.reset_index(inplace = True)
    #Change the proportion into a percentage
    wn_grp["mean"] *= 100
    #Create a new variable that ranks the values based on their percentage
    wn_grp["wn_rank"] = (wn_grp["mean"] / 10).astype(np.int)
    #Specifically change the values with 0 percentage to -1 as a special category i.e no chance it's a real emergency
    wn_grp.loc[wn_grp["mean"] == 0, "wn_rank"] = -1
    
    #Join the london data with the pt_df to match the property type rankings
    df = pd.merge(left = df, right = wn_grp[["ward_name", "wn_rank"]], left_on = "ward_name", right_on="ward_name")
    df["wn_rank"] = df["wn_rank"].astype(np.str)
    
    return df

def station_imputer(df):
    """
    This function imputes the missing stations for the london data set by setting them to the most occurring stations in 
    their ward. 
    
    params:
        df: Dataframe containing the necessary columns
        
    returns:
        df: Dataframe with the missing values imputed
    
    """
    #Create a list of wards that have missing stations
    miss_stat_wards = list(df.loc[df["first_station"].isnull(), "ward_name"].drop_duplicates())
    #Convert the column to a string type. If not, there's a data type mismatch
    #df["first_station"] = df["first_station"].astype(str)
    df["first_station"] = df["first_station"].astype("category")
    #print("Made it here!")
    def ward_to_station(df):
        """
        This function creates a data frame that has the counts of the wards and stations to be used to find the station 
        with the most counts to be used in the imputer function
        """
        station_df = pd.DataFrame(df[["ward_name", "first_station"]].value_counts())
        station_df = station_df.reset_index().rename(columns={0: "Count"})
        
        return station_df
    wts_df = ward_to_station(df=df)
    wts_df.reset_index(drop=True, inplace = True)
    #Loop through all the wards that have missing stations
    for ward in miss_stat_wards:
        #Set a mask based on the ward
        mask = wts_df["ward_name"].str.contains(ward)
        #Create a temporary dataframe containing only the single ward and possibly multiple stations
        temp_df = wts_df.loc[mask]
        temp_df.reset_index(drop=True, inplace=True)
        #Get the index of the station with the highest count in the ward
        max_cnt_ward = temp_df["Count"].argmax()
        #Get the station
        station = temp_df.loc[max_cnt_ward, "first_station"]
        df.loc[(df["ward_name"] == ward) & (df["first_station"].isnull()), "first_station"] = station
    return df
  
    
def station_rank(df):
    """
    This function ranks the stations based on the percentage of real emergencies.
    
    params:
        df - dataframe with necessary variables
    """
    #Group by property type and summarize by the Emergency variable
    stat_grp = df.groupby("first_station")["Emergency"].aggregate(["mean", "sum"]).sort_values(by="mean", ascending=False)
    stat_grp.reset_index(inplace = True)
    #Change the proportion into a percentage
    stat_grp["mean"] *= 100
    #Create a new variable that ranks the values based on their percentage
    stat_grp["stat_rank"] = (stat_grp["mean"] / 10).astype(np.int)
    df = pd.merge(left = df, right = stat_grp[["first_station", "stat_rank"]], left_on = "first_station", right_on="first_station")
    df["stat_rank"] = df["stat_rank"].astype(np.str)
    return df

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
    df["first_time"] = df.groupby("first_station")["first_time"].\
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
    df["station_pumps"] = df.groupby("first_station")["station_pumps"].\
                                                transform(lambda grp: grp.fillna(np.mean(grp)))
    df["station_pumps"] = (np.round(df["station_pumps"])).astype(str)
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
    df["pumps_attending"] = df.groupby("first_station")["pumps_attending"].\
                                                transform(lambda grp: grp.fillna(np.mean(grp)))
    df["pumps_attending"] = (np.round(df["pumps_attending"])).astype(str)
    return df
    
def clean_london(df, add_emergencies=True, make_month=True, make_hour=True, clean_property=True, rank_property_type=True,
                clean_address=True, clean_boroughs=True, rank_wards=True, impute_station=True, rank_station=True, 
                clean_arriving_time=True, clean_station_pumps=True, clean_pumps_attending=True, verbose=True):
    
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
            
    if rank_property_type:
        df = property_type_rank(df)
        if verbose: print("Sucessfully Cleaned and Ranked Property Type!")
            
    if clean_address:
        merge_address(df)
        if verbose: print("Sucessfully Cleaned Address!")
            
    if clean_boroughs:
        drop_boroughs(df)
        if verbose: print("Sucessfully Cleaned Borough Names!")
            
    if rank_wards:
        df = ward_name_rank(df)
        if verbose: print("Sucessfully Cleaned and Ranked Ward Names!")
            
    if impute_station:
        df = station_imputer(df)
        if verbose: print("Sucessfully Imputed Station Names!")
            
    if rank_station:
        df = station_rank(df)
        if verbose: print("Sucessfully Ranked Stations!")
            
    if clean_arriving_time:
        df = arriving_time(df)
        if verbose: print("Sucessfully Cleaned Station Arriving Time!")
            
    if clean_station_pumps:
        df = station_pumps(df)
        if verbose: print("Sucessfully Cleaned Number of Stations with Pumps!")
            
    if clean_pumps_attending:
        df = pumps_attending(df)
        if verbose: print("Sucessfully Cleaned Number of Pumps Attending!")
            
    df = df.drop(["incident_number", "incident_group", "timestamp_of_call", "property_type", "ward_name",
             "first_station"], axis = 1, inplace = False)
    if verbose: print("Sucessfully Dropped Unecessary Columns!")
    
    if verbose: print("\nSucessfully Cleaned London Data!")
        
    return df
    
        
    
