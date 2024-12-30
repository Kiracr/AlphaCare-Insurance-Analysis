import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns


def data_loder():
    return pd.read_csv("../data/Alpha.csv")

def overall_missing_value(df):
    # find total missing value
    Total_missing_values= df.isna().sum().sum()

    # find total NAN value cells
    Total_null_cells = np.prod(df.shape)

    # comput the percentage of missing values
    percnetage_missing = (Total_missing_values / Total_null_cells) * 100

    # prints the rounded result into two decimal places
    print(f"The dataset has {round(percnetage_missing,2)}% missing values")


def missing_values_for_specfic_column(df):
    """"Check for missing values for specfic column"""
    missing_values = df.isnull().sum()
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    column_data_types = df.dtypes
    missing_table = pd.concat([missing_values,missing_percentage,column_data_types], axis = 1, keys=['Missing Values', '% of Total Values', 'Data Type'] )
    return missing_table.sort_values('% of Total Values', ascending=False).round(2)


def drop_high_missing_columns(df):
    missing_values_in_percnetage =  (df.isnull().sum() / len(df)) * 100
    droping_columns = missing_values_in_percnetage[missing_values_in_percnetage > 50].index
    df_cleaned =  df.drop(columns = droping_columns)
    print(f"The droped columns are: {list(droping_columns)} ")

    return df_cleaned

def impute_missing_values(df):
    """
    Impute missing values: mode for categorical, median for numerical columns.
    
    :param df: pandas DataFrame
    :return: DataFrame with imputed values
    """
    for column in df.columns:
        if df[column].dtype == 'object':
            # Categorical column: impute with mode
            mode_value = df[column].mode()[0]
            #df[column].fillna(mode_value)
            df[column] = df[column].fillna(mode_value)

        else:
            skewness = df[column].skew()
            
            if abs(skewness) < 0.5:
                mean = df[column].mean()
                df[column] = df[column].fillna(mean)
            else:
                # Numerical column: impute with median
                median_value = df[column].median()
                #df[column].fillna(median_value)
                df[column] = df[column].fillna(median_value)

    return df
def cap_outliers(df, columns=None):
    """
    Cap outliers in specified numeric columns using the IQR method.
    
    :param df: pandas DataFrame
    :param columns: list of column names to process (if None, all numeric columns will be processed)
    :return: DataFrame with capped outliers
    """
    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    df_capped = df.copy()
    
    if columns is None:
        columns = df_capped.select_dtypes(include=[np.number]).columns
    
    for column in columns:
        Q1 = df_capped[column].quantile(0.25)
        Q3 = df_capped[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df_capped.loc[df_capped[column] < lower_bound, column] = lower_bound
        df_capped.loc[df_capped[column] > upper_bound, column] = upper_bound
    
    return df_capped  


def outlier_box_plots(df):
    for column in df:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=df[column])
        plt.title(f'Box plot of {column}')
        plt.show()                   

    