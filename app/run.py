import json
import plotly
import pandas as pd
import re

from collections import Counter
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    
    '''
    Input:
    text: string. text containing a message.
    
    Output:
    clean_tokens: list of strings. A list of strings containing cleaned and lemmatized tokens.
    
    Workflow:
    - Tokenize by splitting text up into words
    - Lemmatize to reduce words to the root or stem form so they can be analysed as a single item, identified by the word's lemma, or dictionary form.
    - Converting to lowercase and removing punctuation
    '''
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('data/DisasterResponse.db', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    #Figure 2
    category_count = df.iloc[:,4:].sum(axis=0).sort_values(ascending=False)
    category_names = list(category_count.index)
    
    #Figure 3
    data_set = ""
    for i,j in df.iterrows():
        data_set+=j["message"]
    
    # Normalize text by changing to lower case and removing punctuation 
    text = re.sub(r"[^a-zA-Z0-9]",  " ",data_set)
    text = text.lower().strip()
    # split() returns list of words in the text 
    word_list = text.split() 
  
    # Python Counter is a container that will hold the count of each of 
    # the elements present in the container. The counter is a sub-class available inside the dictionary class. 
    container = Counter(word_list) 
  
    # Return a list of the n most common elements and their counts from the most common to the least. 
    # input values and their respective counts. 
    most_common_words  = container.most_common(100) 
    most_common_clean = [w for w in most_common_words if w[0] not in stopwords.words("english")]
    
    x_val = [x[0] for x in most_common_clean[:10]]
    y_val = [x[1] for x in most_common_clean[:10]]
    
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
                    'title': "Frequency"
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
                    y=category_count
                )
            ],
            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Frequency"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=x_val,
                    y=y_val
                )
            ],

            'layout': {
                'title': 'Distribution of Words in Emergency Messages (Top 10)',
                'yaxis': {
                    'title': "Frequency"
                },
                'xaxis': {
                    'title': "Word"
                }
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
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()