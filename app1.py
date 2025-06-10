from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import difflib


data = pd.read_csv('bigmovies.csv', usecols=['id', 'title', 'backdrop_path', 'overview', 'tagline', 'genres', 'keywords'])
data.dropna(subset=['title', 'genres', 'overview', 'tagline', 'keywords', 'backdrop_path'], inplace=True)

data['combined_data'] = data['title'] + ' ' + data['genres'] + ' ' + data['overview'] + ' ' + data['tagline'] + ' ' + data['keywords']

vectorizer = TfidfVectorizer(stop_words='english')
feature_set = vectorizer.fit_transform(data['combined_data'])

nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
nn_model.fit(feature_set)

app = Flask(__name__)

@app.route('/')
def index():
    # Show top 40 movies for homepage
    movies1 = data.head(40)
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
    movie_index = data[data['title'] == match_title].index[0]
    distances, indices = nn_model.kneighbors(feature_set[movie_index], n_neighbors=6)

    results = []
    for i in indices.flatten()[1:]:
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
        return render_template('recommendation1.html', message="Please enter a movie name.")

    results = recommend(user_input)

    if not results:
        return render_template('recommendation1.html', message="No matching movie found. Try a different name.")

    return render_template('recommendation1.html', results=results)

if __name__ == '__main__':
    app.run(port=8001
            , debug=True)
