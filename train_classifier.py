import sys
import re
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    import pandas as pd
    from sqlalchemy import create_engine
    import re
    import nltk
    nltk.download('punkt')
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('df',engine)
    X = df['message']
    y = df[df.columns[4:]]
    return X, y, df.columns[4:]

def tokenize(text):
    from nltk.tokenize import word_tokenize
    from nltk.tokenize import sent_tokenize
    lower = text.lower()
    lower = re.sub(r"[^a-zA-Z0-9]", " ", lower)
    return word_tokenize(lower)


def build_model():
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.linear_model import LogisticRegression
    from lightgbm import LGBMClassifier
    import numpy as np

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LGBMClassifier()))])

    parameters = {
        'clf__estimator__max_depth':np.round(np.linspace(-1,10,3)).astype(int),
        'clf__estimator__n_estimators':[50,100,500]
}

    cv = GridSearchCV(pipeline, param_grid=parameters,scoring='f1_weighted',cv=2,verbose=3)
    
    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    from sklearn.metrics import classification_report
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test,Y_pred,target_names=category_names))


def save_model(model, model_filepath):
    import pickle
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.linear_model import LogisticRegression
    from lightgbm import LGBMClassifier
    import numpy as np
    import pandas as pd
    from sqlalchemy import create_engine
    import re
    import nltk
    nltk.download('punkt')


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
