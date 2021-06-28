import pandas as pd
import numpy as np
import os
from env import host, user, password

###################### Acquire Telco_Churn Data ######################

def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

###################################

#######################################################################

def zillow_data_all():
    '''
    This function reads the zillow data filtered by single family residential from the Codeup db into a df,
    write it to a csv file, and returns the df.
    '''
    # Create SQL query.
    sql_query = '''
    select * from properties_2017 as prop17
    left join (select parcelid, logerror, max(transactiondate) as transactiondate
    FROM predictions_2017 
    group by parcelid, logerror) as pred_2017 ON prop17.parcelid= pred_2017.parcelid
    left join propertylandusetype as propland ON prop17.propertylandusetypeid = propland.propertylandusetypeid
    left join airconditioningtype as actype ON prop17.airconditioningtypeid = actype.airconditioningtypeid
    left join architecturalstyletype as arch ON prop17.architecturalstyletypeid = arch.architecturalstyletypeid
    left join buildingclasstype as bc ON prop17.buildingclasstypeid = bc.buildingclasstypeid
    left join heatingorsystemtype as heat ON prop17.heatingorsystemtypeid = heat.heatingorsystemtypeid
    left join buildingclasstype as bct ON prop17.buildingclasstypeid = bct.buildingclasstypeid
    left join storytype as st ON prop17.storytypeid = st.storytypeid
    left join typeconstructiontype as tc ON prop17.typeconstructiontypeid = tc.typeconstructiontypeid
    where (pred_2017.transactiondate Like '%2017%') and (prop17.longitude is not NULL) and (prop17.latitude is not NULL) and prop17.propertylandusetypeid IN (260, 261,262,263,264,265,268,273,274,275,276, 279);
    '''
    
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('zillow'))
    
    return df



def get_zillow_cached(cached=False):
    '''
    This function reads in zillow data from Codeup database and writes data to
    a csv file if cached == False or if cached == True reads in telco_churn df from
    a csv file, returns df.
    '''
    if cached == False or os.path.isfile('zillow_data.csv') == False:
        
        # Read fresh data from db into a DataFrame.
        df = zillow_data_all()
        
        # Write DataFrame to a csv file.
        df.to_csv('zillow_data.csv')
        
    else:
        
        # If csv file exists or cached == True, read in data from csv file.
        df = pd.read_csv('zillow_data.csv', index_col=0)
        
    return df