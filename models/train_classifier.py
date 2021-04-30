import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import pickle
import re
import time

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import precision_recall_fscore_support,accuracy_score,label_ranking_average_precision_score
from sklearn.base import BaseEstimator, TransformerMixin


def load_data(database_filepath):
    '''
    Load messages and coresponding categories data from sqlite database 
    and return the data in array format for further processing in a 
    supervised machine learning model
    
    Input:
    database_filepath: path where database is stored
    Output:
    X:        numpy array (1D) of emergency text messages
    y:        numpy array (2D) of count matrix of labels (i.e. categories of text messages)
    category: list of strings of category names
    
    '''
    engine = create_engine('sqlite:///'+database_filepath)   
    df = pd.read_sql_table(database_filepath, con=engine)
    
    # define features and label arrays
    X = df.message.values             # turn df column "'message' into numpy.ndarray
    y = df[df.columns[4:]].values     # turn df into numpy.ndarray
    category = list(df.columns[4:])   # turn column names into list
    
    return X, y, category


def tokenize(text):
    '''
    Input:
    text: string. text containing a message.
    
    Output:
    clean_tokens: list of strings. A list of strings containing normalized and lemmatized tokens.
    
    Workflow:
    - Normalize text by converting to lowercase and removing punctuation
    - Tokenize by splitting text up into words
    - Remove Stop words
    - Lemmatize to reduce words to the root or stem form
    '''
    
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words("english")]
    
    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # remove leading/trailing white space
        clean_tok = tok.strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Create a scikit-learn Pipeline with GridSearchCV to output a final model 
    that predicts a message classifications for the 36 categories (multi-output classification)
    use GridSearchCV to exhaustive search over specified parameter values for estimator
    
    Output: cross-validation generator 
    '''
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier(), n_jobs=-1))     # The number of jobs to run in parallel
                                                                            # -1 means using all processors
    ])

    
    parameters = {
        'vect__max_features': (None, 10000),          # If not None, build a vocabulary that only consider the top 
                                                      # max_features ordered by term frequency across the corpus. 
                                                      # (default=None)
                                                      
        
        'clf__estimator__n_estimators': [50,100]      # The maximum number of estimators at which boosting is terminated. 
                                                      # In case of perfect fit, the learning procedure is stopped early. 
                                                      # (default=50 for AdaBoost and default=100 for Random Forests)
    }
                                                      
    cv = GridSearchCV(pipeline, param_grid=parameters,verbose = 3, n_jobs=-1, scoring='f1_samples')
                                                      # # verbose = 3 to get the detailed realtime progress of the operation
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate the model performances and print classification_report
    
    Input:
    - model: sklearn model to use for predicting labels
    - X_test (pandas.Series): messages for which we want to predict categories 
    - Y_test (pandas.DataFrame): dataframe containing the categories of X_test
    - category_names: (str): categories name
    Output: 
    - print statement of classification_report
    '''
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))

    

def save_model(model, model_filepath):
    '''
    Save the model using pickle
    Input:
    - model: model you want to save
    - model_filepath: where you want to save the model
    No Output
    '''
    pkl_filename = model_filepath
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()