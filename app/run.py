import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar, Heatmap
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)


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


# load data
engine = create_engine('sqlite:///./data/DisasterResponse.db')
df = pd.read_sql_table('Messages', engine)

# load model
model = joblib.load("./models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_counts_df = df.iloc[:, 4:].sum().sort_values(ascending=False)
    category_names = category_counts_df.index
    category_counts = category_counts_df.values

    categories_sorted = df.iloc[:, 5:].sum().sort_values(ascending=False).\
        index.values
    categories_top = df.iloc[:, 5:][categories_sorted]

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Sorted Count of Messsage Categories',
                'yaxis': {
                    'title': "Count"
                },
            }
        },
        {
            'data': [
                Heatmap(
                    x=categories_sorted,
                    y=categories_sorted,
                    z=categories_top.corr().values
                )
            ],

            'layout': {
                'title': 'Correlation Between Categories'
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='127.0.0.1', port=5000, debug=True)


if __name__ == '__main__':
    main()
