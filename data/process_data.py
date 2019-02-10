# Import Statements
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Load and merge CSV data

    Input:
        messages_filepath - Path of disaster_messages CSV file
        categories_filepath - Path of disaster_categories CSV file

    Output:
        mes_and_cat - DataFrame of merged messages and categories files
    '''

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    mes_and_cat = messages.merge(categories, how='outer', on='id')

    return mes_and_cat


def clean_data(df):
    '''
    Clean and return the DataFrame df

    Input:
        df - DataFrame to apply cleaning to

    Output:
        df_clean - DataFrame of clean df
    '''

    # Split the categories column further by ';'
    categories_split = df['categories'].str.split(pat=';', expand=True)

    # Take the first element to get the row_names from categories_split
    row_names = categories_split.iloc[0]

    # Split out the row_names by taking the
    categories_names = row_names.apply(lambda x: x[:-2])

    df_clean = pd.concat([df, pd.DataFrame(columns=categories_names)], axis=1)

    # Encode category values to 0/1
    for col in categories_names:
        df_clean[col] = df_clean['categories'].apply(lambda x: 0
                                                     if x.find(col + '-1')
                                                     == -1 else 1)

    # Drop the original labels
    df_clean.drop(inplace=True, labels='categories', axis=1)

    # Drop duplicate rows, keeping the first occurance
    df_clean.drop_duplicates(inplace=True, keep='first')

    return df_clean


def save_data(df, database_filename):
    '''
    Save a DataFrame to an SQLLite Database

    Input:
        df - DataFrame to be saved
        database_filename - Path to save SQLLite Database to
    Output:
        None
    '''

    engine = create_engine('sqllite:///' + database_filename)
    df.to_sql('Messages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = \
            sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
