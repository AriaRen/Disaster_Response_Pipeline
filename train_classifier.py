# import libraries and packages
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
nltk.download('stopwords')
import re
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier


def load_data(database_filepath):
    '''
    INPUT
    data_filepath - a string representing file path from where we can load data
    
    OUTPUT
    X, y - python pandas series or dataframe representing X and y used for modeling
    '''
    engine = create_engine('sqlite:///../data/DisasterResponse.db')
    df = pd.read_sql("SELECT * FROM msg", engine)
    X = df.message.values
    y = df[df.columns[4:]].values
    return X, y


def tokenize(text):
    '''
    INPUT
    text - a string to be tokenized and cleaned for model purpose
    
    OUTPUT
    clean_tokens - a list containing cleaned words and elements in the string to be used for modeling
    '''

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_url = re.findall(url_regex, text)
    for url in detected_url:
        text = text.replace(url, "urlplaceholder")
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if clean_tok not in ":,.-(){}<>'s'm ":
            if clean_tok not in stopwords.words("english"):
                clean_tokens.append(clean_tok)
    
    return clean_tokens

# class StartingVerbExtractor(BaseEstimator, TransformerMixin):

#     def starting_verb(self, text):
#         sentence_list = nltk.sent_tokenize(text)
#         for sentence in sentence_list:
#             pos_tags = nltk.pos_tag(tokenize(sentence))
#             try:
#                 first_word, first_tag = pos_tags[0]
#                 if first_tag in ['VB', 'VBP']:
#                     return True
#             except:
#                 return False
#         return False

#     def fit(self, x, y=None):
#         return self

#     def transform(self, X):
#         X_tagged = pd.Series(X).apply(self.starting_verb)
#         return pd.DataFrame(X_tagged)


def build_model():
    '''
    INPUT
    Nothing
    
    OUTPUT
    cv - a python scikit learn model after pipeline and grid search 
    '''
    pipeline = Pipeline([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
    
    parameters = {
    'text_pipeline__vect__ngram_range': [(1, 1), (1, 2)],
    'clf__estimator__n_estimators': [10, 20],
    'clf__estimator__min_samples_split': [2, 3, 4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)

    return cv
    


def display_results(cv, Y_test, Y_pred):
    '''
    INPUT
    cv - a python scikit learn model after pipeline and grid search 
    Y_test - a pandas series representing existed test Y data
    Y_pred - a pandas series representing numbers predicted by the model we just created 

    OUTPUT
    Accuracy - a float representing accuracy rate to demonstrate model performance 
    Best Parameters - a list containing best parameters resulted from grid search 
    '''
    #labels = np.unique(Y_pred)
    #confusion_mat = confusion_matrix(Y_test, Y_pred, labels=labels)
    accuracy = (Y_pred == Y_test).mean()

    #print("Labels:", labels)
    #print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)
    print("\nBest Parameters:", cv.best_params_)



def save_model(model, model_filepath):
    '''
    INPUT
    cv - a python scikit learn model after pipeline and grid search 
    Y_test - a pandas series representing existed test Y data
    Y_pred - a pandas series representing numbers predicted by the model we just created 

    OUTPUT
    Accuracy - a float representing accuracy rate to demonstrate model performance 
    Best Parameters - a list containing best parameters resulted from grid search 
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Predict model...')
        Y_pred = model.predict(X_test)
        
        print('Evaluating model...')
        display_results(model, Y_test, Y_pred)

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