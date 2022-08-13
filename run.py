import json
import plotly
import pandas as pd



from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import plotly.express as px
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///DisasterResponse.db')
df = pd.read_sql_table('df', engine)

# load model
model = joblib.load("classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    count_df = df.drop(columns={'id','message','original','genre'})
    number = count_df.sum()
    count_df2 = pd.DataFrame()

    count_df2['Subject'] = number.index
    count_df2['Number'] = list(number)
    count_df2.sort_values(by='Number',ascending=False,inplace=True)

    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    word_list = list(df['message'])
    unique_string=(" ").join(word_list)

    wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)
    wordcloud_fig = px.imshow(wordcloud)
    wordcloud_fig.update_layout(
        title=dict(text='Most Common Words in the Messages',x=0.5),
        yaxis={'showgrid':False,'showticklabels':False,'zeroline':False},
        xaxis={'showgrid':False,'showticklabels':False,'zeroline':False},
        hovermode=False)



    graphs = [


        {'data': [
            Bar(
                x=count_df2['Subject'],
                y=count_df2['Number'])
        ],
         'layout': {
                'title': 'Number of Messages By Subject',
                'yaxis': {
                    'title': "Number"
                },
                'xaxis': {
                    'title': "Subject"
                }
            }},

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
        }
    ]

    graphs.append(wordcloud_fig)



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
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
