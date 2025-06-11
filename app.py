from flask import Flask, render_template, request
import pickle

import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

data = pickle.load(open('../bigmovies.pkl', 'rb'))
feature_set = pickle.load(open('../features.pkl', 'rb'))
nn_model = pickle.load(open('../model.pkl', 'rb'))

movies1 = data.head(40)

app = Flask(__name__)

@app.route('/')
def index():
    movie_data = list(zip(
        movies1['title'],
        movies1['backdrop_path'],
        movies1['genres'],
        movies1['overview']
    ))
    return render_template('index1.html', movie_data=movie_data)

@app.route('/recommend')
def recommend_ui():
    return render_template('recommendation1.html')

def recommend(user_input):
    user_input = user_input.strip().lower()
    list_of_titles = data['title'].tolist()
    close_match = difflib.get_close_matches(user_input, list_of_titles)

    if not close_match:
        return []

    match_title = close_match[0]
    match_index = data[data['title'] == match_title].index[0]

    distances, indices = nn_model.kneighbors(feature_set[match_index], n_neighbors=20)

    results = []
    for i in indices.flatten()[1:]:  # Skip the queried movie itself
        temp = data.iloc[i]
        results.append([
            temp['title'],
            temp['genres'],
            temp['overview'],
            temp['backdrop_path']
        ])
    return results

@app.route('/recommend_movies', methods=['POST'])
def recommend_route():
    user_input = request.form.get('user_input')

    if not user_input:
        return render_template("recommendation1.html", message="Please enter a movie name.",user_input=user_input)

    results = recommend(user_input)

    if not results:
        return render_template("recommendation1.html", message="No matching movie found. Try another.",user_input=user_input)

    return render_template("recommendation1.html", results=results,user_input=user_input)


@app.route('/search', methods=['GET', 'POST'])
def search():

    query      = ""
    results    = []

    if request.method == 'POST':
        query = request.form.get('query', '').strip().lower()

        if query:
            possible_titles = data['title'].tolist()
            close_matches   = difflib.get_close_matches(query,possible_titles,n=5)

            for t in close_matches:
                row = data[data['title'] == t].iloc[0]
                results.append([
                    row['title'],
                    row['genres'],
                    row['overview'],
                    row['backdrop_path']
                ])

    return render_template('search.html',query=query,results=results)





if __name__ == '__main__':
    app.run(port=8000)
