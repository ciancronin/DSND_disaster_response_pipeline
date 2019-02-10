import sys
import pandas as pd
import pickle

from sqlalchemy import create_engine

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    '''
    Load data from an SQLLite Database

    Input:
        database_filepath - Path of the SQLLite Database to load

    Output:
        X - Series of messages
        Y - DataFrame of categories
        Y_categories - Numpy array of category names
    '''

    engine = create_engine('sqllite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)
    X = df['message']
    Y = df.drop(labels=['id', 'message', 'original', 'genre'])
    Y_categories = Y.columns.values

    return X, Y, Y_categories


def tokenize(text):
    '''
    Transform text into tokens, and clean them

    Input:
        text - line of text to tokenize
    Output:
        tokens_clean - List of clean tokens
    '''

    # Tokenize text
    tokens = word_tokenize(text)

    # Initiate Lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Iterate through each token and clean
    tokens_clean = []

    for token in tokens:
        token_clean = lemmatizer.lemmatize(token).lower().strip()
        tokens_clean.appen(token_clean)

    return tokens_clean


def build_model():
    '''
    Build multiclass classification pipeline

    Input:
        None
    Output:
        cv - GridSearchCV object of optimal model parameters
    '''

    # Pipeline dict
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier())),
        ])

    # Parameters dict for GridSearchCV
    parameters = {
        'tfidf__use_idf': [False, True],
        'clf__n_estimators': [50, 100],
        'clf__min_samples_split': [2, 4, 8]
        }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Print the performance metrics of our classification model

    Input:
        model - Classification model
        X_test - Text input from the test data
        Y_test - Labels from the test data
        category_names - List of output labels
    Output:
        None
    '''

    Y_pred = pd.DataFrame(model.predict(X_test), index=Y_test.index.values,
                          columns=category_names)

    cat_perf = []

    # TODO Ensure category names is correct when running tests
    for cat in category_names:
        print(classification_report(Y_test[cat], Y_pred[cat]))

        cat_accuracy = (Y_test[cat] == Y_pred[cat]).mean()
        cat_precision = precision_score(Y_test[cat], Y_pred[cat])
        cat_recall = recall_score(Y_test[cat], Y_pred[cat])
        cat_f1_score = f1_score(Y_test[cat], Y_pred[cat])

        cat_perf.append([cat_accuracy,
                        cat_precision,
                        cat_recall,
                        cat_f1_score])

    performance_output = pd.DataFrame(cat_perf, index=Y_test.columns.values,
                                      columns=['accuracy',
                                               'precision',
                                               'recall',
                                               'f1_score'])

    print(performance_output)


def save_model(model, model_filepath):
    '''
    Save the model for use in our Python Flask App using Pickle

    Input:
        model - Classification model
        model_filepath - Path to save model to
    Output:
        None
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                            test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
