import sys
# import libraries
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
    engine = create_engine('sqlite:///InsertDatabaseName.db')
    df = pd.read_sql("SELECT * FROM msg_category", engine)
    X = df.message.values
    y = df[df.columns[4:]].values
    return X, y


def tokenize(text):
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

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            try:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP']:
                    return True
            except:
                return False
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model():
    pipeline = Pipeline([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
    
    parameters = {
    'text_pipeline__vect__ngram_range': [(1, 1)],
    'clf__estimator__n_estimators': [10]
    #'clf__estimator__min_samples_split':[2, 3],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)

    return cv
    


def display_results(cv, Y_test, Y_pred):
    #labels = np.unique(Y_pred)
    #confusion_mat = confusion_matrix(Y_test, Y_pred, labels=labels)
    accuracy = (Y_pred == Y_test).mean()

    #print("Labels:", labels)
    #print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)
    print("\nBest Parameters:", cv.best_params_)



def save_model(model, model_filepath):
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