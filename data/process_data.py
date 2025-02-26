import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages.merge(categories, how='left')


def clean_data(df):
    categories = df[['categories', 'id']]
    # split column categories by ';' delimiter and put into dataframe
    split_cat = categories['categories'].str.split(';', expand=True)
    row = split_cat.loc[:0]
    category_colnames = [el[0] for el in row.apply(lambda val: val.str.split('-')[0])]
    split_cat.columns = category_colnames

    # create new categories dataframe with expanded categories columns
    categories = categories.append(split_cat, sort=False)
    for col in split_cat:
        categories[col] = pd.to_numeric(split_cat[col].str[-1])

    # merge df with new categories and drop unrelevant columns
    df = df.merge(categories, how='left')
    df = df.drop(['categories', 'related', 'original'], axis=1)
    df_no_dup = df[~df.duplicated()]

    return df_no_dup


def save_data(df, database_filename):
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql(database_filename.replace(".db", ""), engine, index=False, if_exists='replace')


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
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
