import sys
import pandas as pd
from sqlalchemy import create_engine

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
import pickle

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    engine = create_engine(f'sqlite:///../data/{database_filepath}.db')
    df = pd.read_sql(f'{database_filepath}', con=engine)
    X = df['message']
    # drop id because it is not relevant and child_alone, because it only contains zeroes

    y = df.drop(['id',
                 'genre',
                 'message',
                 'child_alone',
                 'offer',
                 'shops',
                 'tools',
                 'fire',
                 'hospitals',
                 'missing_people',
                 'aid_centers'], axis=1)
    return X, y, y.columns


def tokenize(text):
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens


def build_fit_model(X_train, y_train):
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'vect__max_df': [0.5, 0.75, 1.0],
        'vect__max_features': [None, 5000, 10000],
        'tfidf__use_idf': [True, False],
        'clf__estimator__loss': ['hinge', 'squared_hinge'],
        'clf__estimator__alpha': [0.001, 0.01, 0.00001],
    }

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(SGDClassifier(tol=None, max_iter=5)))
    ])

    cv = GridSearchCV(pipeline, param_grid=parameters)
    cv.fit(X_train, y_train)

    return cv


def evaluate_models(model, X_test, y_test, category_names):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    pickle.dump(model, open(f'{model_filepath}/model', 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

        print('Building and training models...')
        model = build_fit_model(X_train, y_train)


        print('Evaluating model...')
        evaluate_models(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
                  'save the model to as the second argument. \n\nExample: python ' \
                  'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
