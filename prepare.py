import seaborn as sns
import os
from pydataset import data
from scipy import stats
import pandas as pd
import numpy as np

#train validate, split
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

# metrics and confusion
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#model classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ignore warnings

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

############## Data Prep #########################

def remove_columns(df, cols_to_remove):  
    df = df.drop(columns=cols_to_remove)
    return df

def handle_missing_values(df, prop_required_column = .9, prop_required_row = .9):
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

def data_prep(df, cols_to_remove=[], prop_required_column=.5, prop_required_row=.75):
    df = remove_columns(df, cols_to_remove)
    df = handle_missing_values(df, prop_required_column, prop_required_row)
    return df


############################################################
#   Standard data examination
###############################
def summarize(df):
    '''
    summarize will take in a single arguement(pandas dataframe)
    and output to console, various statisitics on said datafram.
    Including : .head(), .info(), .describe(),.value_counts(): spread of data
    and observations of nulls in the dataframe.
    '''
    print('===================================================')
    print('Dataframe head: ')
    print(df.head(10))
    print('===================================================')
    print('Dataframe info: ')
    print(df.info())
    print('===================================================')
    print('Dataframe Description: ')
    print(df.describe())
    num_cols = [col for col in df.columns if df[col].dtype != 'O']
    cat_cols = [col for col in df.columns if col not in num_cols]
    print('===================================================')
    print('DataFrame value counts: ')
    for col in df.columns:
        if col in cat_cols:
            print(df[col].value_counts)
        else:
            print(df[col].value_counts(bins=10, sort = False))
    print('===================================================')
    
    
    
    
def view_null_records(df, variable):
    """
    function allows you to records which contain null, nan values.
    REMEMBER, will only work for individual column and if that columns has nulls, 
    otherwise will return empty dataframe
    """
    df[df[variable].isna()]
    
    return df


#######
# missing values
###########################

def miss_dup_values(df):
    '''
    this function takes a dataframe as input and will output metrics for missing values and duplicated rows,
    and the percent of that column that has missing values and duplicated rows
    '''
        # Total missing values
    mis_val = df.isnull().sum()
        # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
        #total of duplicated
    dup = df.duplicated().sum()
        # Percentage of missing values
    dup_percent = 100 * dup / len(df)
        # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
    mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)
        # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
           "There are " + str(mis_val_table_ren_columns.shape[0]) +
           " columns that have missing values.")
    print( "  ")
    print (f"** There are {dup} duplicate rows that represents {round(dup_percent, 2)}% of total Values**")
        # Return the dataframe with missing information
    return mis_val_table_ren_columns

####################################
                #outlier finding###
#####################################

def outlier_bound_calculation(df, variable):
    '''
    calcualtes the lower and upper bound to locate outliers in variables
    '''
    quartile1, quartile3 = np.percentile(df[variable], [25,75])
    IQR_value = quartile3 - quartile1
    lower_bound = quartile1 - (1.5 * IQR_value)
    upper_bound = quartile3 + (1.5 * IQR_value)
    '''
    returns the lowerbound and upperbound values
    '''
    return print(f'For {variable} the lower bound is {lower_bound} and  upper bound is {upper_bound}')


def detect_outliers(df, k, col_list):
    ''' get upper and lower bound for list of columns in a dataframe 
        if desired return that dataframe with the outliers removed
    '''
    
    odf = pd.DataFrame()
    
    for col in col_list:

        q1, q2, q3 = df[f'{col}'].quantile([.25, .5, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound
        
        # print each col and upper and lower bound for each column
        print(f"{col}: Median = {q2} lower_bound = {lower_bound} upper_bound = {upper_bound}")

        # return dataframe of outliers
        odf = odf.append(df[(df[f'{col}'] < lower_bound) | (df[f'{col}'] > upper_bound)])
            
    return odf



def show_outliers(df, k, columns):
    '''
    calculates the lower and upper bound to locate outliers and displays them
    recommended k be 1.5 and entered as integer
    '''
    for i in columns:
        quartile1, quartile3 = np.percentile(df[i], [25,75])
        IQR_value = quartile3 - quartile1
        lower_bound = (quartile1 - (k * IQR_value))
        upper_bound = (quartile3 + (k * IQR_value))
        print(f'For {i} the lower bound is {lower_bound} and  upper bound is {upper_bound}')
        
        
def remove_outliers(df, k, columns):
    '''
    calculates the lower and upper bound to locate outliers in variables and then removes them.
    recommended k be 1.5 and entered as integer
    '''
    for i in columns:
        quartile1, quartile3 = np.percentile(df[i], [25,75])
        IQR_value = quartile3 - quartile1
        lower_bound = (quartile1 - (k * IQR_value))
        upper_bound = (quartile3 + (k * IQR_value))
        print(f'For {i} the lower bound is {lower_bound} and  upper bound is {upper_bound}')
        df = df[(df[i] <= upper_bound) & (df[i] >= lower_bound)]
        print('-----------------')
        print('Dataframe now has ', df.shape[0], 'rows and ', df.shape[1], 'columns')
    return df

def remove_outliers_noprint(df, k, columns):
    '''
    calculates the lower and upper bound to locate outliers in variables and then removes them.
    recommended k be 1.5 and entered as integer
    '''
    for i in columns:
        quartile1, quartile3 = np.percentile(df[i], [25,75])
        IQR_value = quartile3 - quartile1
        lower_bound = (quartile1 - (k * IQR_value))
        upper_bound = (quartile3 + (k * IQR_value))

        df = df[(df[i] <= upper_bound) & (df[i] >= lower_bound)]
        
    return df
        
        
        

###############################
#  nulls in columns and rows#
##############################

def nulls_by_col(df):
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    percent_missing = num_missing / rows * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing':percent_missing})
    # col by col assessment.
    return cols_missing
    
def nulls_by_row(df):
    num_missing = df.isnull().sum(axis =1) # sum of column axis.
    percent_missing = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing:': num_missing, 'percent_cols_missing:':percent_missing})\
    .reset_index()\
    .groupby(['num_cols_missing:', 'percent_cols_missing:']).count()\
    .rename(index = str, columns = {'index': 'num_rows'}).reset_index()
    # creates large data`frame on row by row basis.
    return rows_missing


##############################################################################
# extract objects or numerical based columns 


def get_object_cols(df):
    """
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names.
    """
    # create a mask of columns whether they are object type or not
    mask = np.array(df.dtypes == "object")

    # get a list of the column names that are objects (from the mask)
    object_cols = df.iloc[:, mask].columns.tolist()

    return object_cols


def get_numeric_X_cols(X_train, object_cols):
    """
    takes in a dataframe and list of object column names
    and returns a list of all other columns names, the non-objects.
    """
    numeric_cols = [col for col in X_train.columns.values if col not in object_cols]

    return numeric_cols


###### general splitting##########

# Generic splitting function for continuous target.
def split_continuous(df):
    '''
    Takes in a df
    Returns train, validate, and test DataFrames
    '''
    # Create train_validate and test datasets
    train_validate, test = train_test_split(df, 
                                        test_size=.2, 
                                        random_state=123)
    # Create train and validate datsets
    train, validate = train_test_split(train_validate, 
                                   test_size=.3, 
                                   random_state=123)

    # Take a look at your split datasets

    print(f'train -> {train.shape}')
    print(f'validate -> {validate.shape}')
    print(f'test -> {test.shape}')
    return train, validate, test
        ####################################################################
# Train, validate, split which doesnt exclude target from train. 
# target is categorical 


#genreal split when categorical
def general_split(df, stratify_var):
    '''
    This function take in the telco_churn_data acquired by get_telco_churn,
    performs a split and stratifies total_charges column. Can specify stratify as None which will make this useful for continous.
    Returns train, validate, and test dfs.
    '''
    #20% test, 80% train_validate
    train_validate, test = train_test_split(df, test_size=0.2, 
                                        random_state=1349, stratify = stratify_var)
    # 80% train_validate: 30% validate, 70% train.
    train, validate = train_test_split(train_validate, train_size=0.7, 
                                   random_state=1349, stratify = stratify_var)
    """
    returns train, validate, test 
    """
    return train, validate, test

##################################################
# train, validate, split #
#  generates features and target.
################################################

def train_validate_test(df, target, stratify):
    """
    this function takes in a dataframe and splits it into 3 samples,
    a test, which is 20% of the entire dataframe,
    a validate, which is 24% of the entire dataframe,
    and a train, which is 56% of the entire dataframe.
    It then splits each of the 3 samples into a dataframe with independent variables
    and a series with the dependent, or target variable.
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test.
    """
    # split df into test (20%) and train_validate (80%)
    train_validate, test = train_test_split(df, test_size=0.2, random_state=123, stratify = stratify)

    # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=123, stratify = stratify)

    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target])
    y_train = train[target]

    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]

    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]
    '''    
    Returns X_train, y_train, X_validate, y_validate, X_test, y_test
    '''

    return X_train, y_train, X_validate, y_validate, X_test, y_test




##############################################################################
def model_metrics(X_train, y_train, X_validate, y_validate):
    '''
    this function will score models and provide confusion matrix.
    returns classification report as well as evaluation metrics.
    '''
    lr_model = LogisticRegression(random_state =1349)
    dt_model = DecisionTreeClassifier(max_depth = 2, random_state=1349)
    rf_model = RandomForestClassifier(max_depth=4, min_samples_leaf=3, random_state=1349)
    kn_model = KNeighborsClassifier()
    models = [lr_model, dt_model, rf_model]
    for model in models:
        #fitting our model
        model.fit(X_train, y_train)
        #specifying target and features
        train_target = y_train
        #creating prediction for train and validate
        train_prediction = model.predict(X_train)
        val_target = y_validate
        val_prediction = model.predict(X_validate)
        # evaluation metrics
        TN_t, FP_t, FN_t, TP_t = confusion_matrix(y_train, train_prediction).ravel()
        TN_v, FP_v, FN_v, TP_v = confusion_matrix(y_validate, val_prediction).ravel()
        #calculating true positive rate, false positive rate, true negative rate, false negative rate.
        tpr_t = TP_t/(TP_t+FN_t)
        fpr_t = FP_t/(FP_t+TN_t)
        tnr_t = TN_t/(TN_t+FP_t)
        fnr_t = FN_t/(FN_t+TP_t)
        tpr_v = TP_v/(TP_v+FN_v)
        fpr_v = FP_v/(FP_v+TN_v)
        tnr_v = TN_v/(TN_v+FP_v)
        fnr_v = FN_v/(FN_v+TP_v)
        
        
        
        print('--------------------------')
        print('')
        print(model)
        print('train set')
        print('')
        print(f'train accuracy: {model.score(X_train, y_train):.2%}')
        print('classification report:')
        print(classification_report(train_target, train_prediction))
        print('')
        print(f'''
        True Positive Rate:{tpr_t:.2%},  
        False Positive Rate :{fpr_t:.2%},
        True Negative Rate: {tnr_t:.2%},  
        False Negative Rate: {fnr_t:.2%}''')
        print('------------------------')
        
        print('validate set')
        print('')
        print(f'validate accuracy: {model.score(X_validate, y_validate):.2%}')
        print('classification report:')
        print(classification_report(y_validate, val_prediction))
        print('')
        print(f'''
        True Positive Rate:{tpr_v:.2%},  
        False Positive Rate :{fpr_v:.2%},
        True Negative Rate: {tnr_v:.2%},  
        False Negative Rate: {fnr_v:.2%}''')
        print('')
        print('------------------------')
####################################################################





########################################
#                scaling
########################################

def Min_Max_Scaler(X_train, X_validate, X_test, target):
    """
    Takes in X_train, X_validate and X_test dfs with numeric values only
    Returns scaler, X_train_scaled, X_validate_scaled, X_test_scaled dfs 
    """
    scaler = sklearn.preprocessing.MinMaxScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), index = X_train.index, columns = X_train.columns)
    X_validate_scaled = pd.DataFrame(scaler.transform(X_validate), index = X_validate.index, columns = X_validate.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index = X_test.index, columns = X_test.columns)
    
    return X_train_scaled, X_validate_scaled, X_test_scaled
        
    
def train_validate_test_scaled (train, validate, test, target):
    """
    Takes in X_train, X_validate and X_test dfs
    Returns X_train_scaled, y_train_scaled, X_validate_scaled, y_validate_scaled, X_test_scaled, y_test_scaled dfs 
    """
    
    y_train = train[target]
    X_train = train.drop(columns = target)
    

    y_validate = validate[target] 
    X_validate = validate.drop(columns = target)
    
    
    y_test = test[target]
    X_test = test.drop(columns = target)
    
    
    #### scale and create scaled features and target
    scaler = MinMaxScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), index = X_train.index, columns = X_train.columns)
    y_train_scaled = pd.DataFrame(scaler.transform(y_train))
    
    
    X_validate_scaled = pd.DataFrame(scaler.transform(X_validate), index = X_validate.index, columns = X_validate.columns)
    y_validate_scaled = pd.DataFrame(scaler.transform(y_validate), index = y_validate.index)
    
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index = X_test.index, columns = X_test.columns)
    y_test_scaled = pd.DataFrame(scaler.transform(y_test), index = y_test.index)
    
    return X_train_scaled, y_test_scaled, X_validate_scaled, y_validate_scaled, X_test_scaled, y_test_scaled



def min_max_scale(X_train, X_validate, X_test, numeric_cols):
    """
    this function takes in 3 dataframes with the same columns,
    a list of numeric column names (because the scaler can only work with numeric columns),
    and fits a min-max scaler to the first dataframe and transforms all
    3 dataframes using that scaler.
    it returns 3 dataframes with the same column names and scaled values.
    """
    # create the scaler object and fit it to X_train (i.e. identify min and max)
    # if copy = false, inplace row normalization happens and avoids a copy (if the input is already a numpy array).

    scaler = MinMaxScaler(copy=True).fit(X_train[numeric_cols])

    # scale X_train, X_validate, X_test using the mins and maxes stored in the scaler derived from X_train.
    #
    X_train_scaled_array = scaler.transform(X_train[numeric_cols])
    X_validate_scaled_array = scaler.transform(X_validate[numeric_cols])
    X_test_scaled_array = scaler.transform(X_test[numeric_cols])

    # convert arrays to dataframes
    X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=numeric_cols).set_index(
        [X_train.index.values]
    )

    X_validate_scaled = pd.DataFrame(
        X_validate_scaled_array, columns=numeric_cols
    ).set_index([X_validate.index.values])

    X_test_scaled = pd.DataFrame(X_test_scaled_array, columns=numeric_cols).set_index(
        [X_test.index.values]
    )
    
    """
    returns X_train_scaled, X_validate_scaled, X_test_scaled 
    """

    return X_train_scaled, X_validate_scaled, X_test_scaled

#######################
def encoding(df, cols, drop_first=True):
    '''
    Take in df and list of columns
    add encoded columns derived from columns in list to the df
    '''
    for col in cols:

        dummies = pd.get_dummies(df[f'{col}'], drop_first=drop_first) # get dummy columns

        df = pd.concat([df, dummies], axis=1) # add dummy columns to df
        
    return df


##### Prep Based on Dataset ##################
 
def prep_zillow(df):
    '''
    This function take in the zillow data acquired by acquire functintion,
    Returns prepped train, validate, and test dfs with outliers removed, Null values dropped,
    and some addtional columns created.
    '''
    df= df.loc[:, ~df.columns.duplicated()]
    df.set_index('parcelid', inplace = True)
    
    df = handle_missing_values(df)
    #dropping nan values
    df.dropna(inplace = True )
    df = remove_columns(df,['propertylandusedesc','finishedsquarefeet12', 'censustractandblock','propertycountylandusecode', 
                                    'rawcensustractandblock', 'assessmentyear','id', 'regionidcity', 'roomcnt','propertylandusetypeid','calculatedbathnbr'])
    df.rename(columns = {'calculatedfinishedsquarefeet':'total_square_ft'}, inplace = True)
    df['acres'] = df.lotsizesquarefeet/43560
    df['structure_dollar_per_sqft'] = df.structuretaxvaluedollarcnt/df.total_square_ft
    df['land_dollar_per_sqft'] = df.landtaxvaluedollarcnt/df.lotsizesquarefeet
    df['county_name'] = df['fips'].map({6037:'Los_Angeles', 6059:'Orange', 6111:'Ventura'})
    df['house_age'] = (2017-df.yearbuilt)
    #dropping columns
    df.drop(columns = ['regionidzip', 'fips', 'fullbathcnt','transactiondate','regionidcounty','yearbuilt','taxvaluedollarcnt','lotsizesquarefeet',
                          'structuretaxvaluedollarcnt','landtaxvaluedollarcnt','taxamount'], inplace = True)   
    df['total_square_ft'] = df['total_square_ft'].astype(int)
    df['structure_dollar_per_sqft'] = df['structure_dollar_per_sqft'].astype(int)
    df['land_dollar_per_sqft'] = df['land_dollar_per_sqft'].astype(int)
    df['house_age'] = df['house_age'].astype(int)
    
    
    # dropping null values.
    df.dropna(inplace = True)

    
    return df

    
    
    