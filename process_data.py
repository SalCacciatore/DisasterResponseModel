import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Loads the data from the csv files and merges them into a single dataframe

    Inputs:
    messages_filepath: filepath to the messages csv file
    categories_filepath: filepath to the categories csc files

    Returns:
    df: merged dataframe
    '''

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,on='id')
    return df

def clean_data(df):
    '''
    Takes the merged dataframe and cleans it make it usable for further analysis.

    Input:
    df: merged dataframe

    Returns:
    df: cleaned dataframe


    '''


    categories = df['categories'].str.split(';',expand=True)
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x:x[-1])

    # convert column from string to numeric
        categories[column] = categories[column].apply(lambda x:int(x))
        categories = categories[categories['related']!=2]
    df.drop('categories',axis=1,inplace=True)
    df = df.merge(categories,right_index=True,left_index=True)
    df.drop_duplicates(inplace=True)
    df.related.replace(2,1,inplace=True)
    return df


def save_data(df, database_filename):
    '''
    Takes the cleaned dataframe and converts it into a SQL database.

    Inputs:
    df: cleaned dataframe
    database_filename: name of the SQL database_filepath

    


    '''

    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('df', engine, index=False, if_exists='replace')


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
