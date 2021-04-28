import sys
import pandas as pd
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load the data from the input files
    Input:
        categories_filename (str): categories filename
        messages_filename (str): messages filename
    Returns:
        df (pandas.DataFrame): dataframe containing the uncleaned dataset
    '''
    messages   = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    messages.drop_duplicates(inplace=True)
    categories.drop_duplicates(inplace=True, subset='id') #some ids have multiple category entries, keep only the first one
    
    
    # Merge the messages and categories datasets using the common id
    # Assign this combined dataset to df, which will be cleaned in the following steps
    df = messages.merge(categories, left_on = 'id', right_on = 'id', how = 'inner')
    
    return df


def clean_data(df):
    '''
    Clean the data
    Input:
        df (pandas.DataFrame): dataframe containing the uncleaned dataset
    Returns:
        df (pandas.DataFrame): dataframe containing the cleaned dataset
    '''
    
    # Split the values in the categories column on the ; character so that each value becomes a separate column. 
    categories = df['categories'].str.split(';', expand=True)
    
    # Use the first row of categories dataframe to create column names for the categories data.
    row = categories.iloc[0,:]
    # Use only the last 2 characters of each string with slicing
    category_colnames = row.apply(lambda n: n[:-2]) 
    
    # Rename columns of categories with new column names.
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1.
    # Iterate through the category columns in df to keep only the last character of each string (the 1 or 0). 
    # For example, related-0 becomes 0, related-1 becomes 1. 
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])

        # data cleaning, replace 2s with 1s
        categories[column] = categories[column].str.replace('2','1')
    
        # Convert the string to a numeric value.
        categories[column] = pd.to_numeric(categories[column])
        
    # Drop the categories column from the df dataframe since it is no longer needed.
    df.drop('categories', axis=1, inplace=True)
    
    # Concatenate df and categories dataframes.
    df = pd.concat([df,categories], axis=1)
    
    # remove duplicates
    df.drop_duplicates(inplace=True)
    
    return df
    


def save_data(df, database_filename):
    '''
    Save the data into a sql database. 
    Input:
        df (pandas.DataFrame): dataframe containing the dataset
        database_filename (str): database filename
    Returns:
        none
    '''
    
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql(database_filename, engine, index=False, if_exists='replace')
    
    return  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()