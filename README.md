# 🎬 Movie Recommendation ML

A machine learning-powered movie recommendation system built with a user-friendly web interface using Streamlit.

---

## 📌 Table of Contents

* [Features](#features)
* [Concept](#concept)
* [How It Works](#how-it-works)
* [Installation](#installation)
* [Usage](#usage)
* [Repository Structure](#repository-structure)
* [Dependencies](#dependencies)
* [Contributing](#contributing)

---

## 🚀 Features

* Recommends similar movies based on content (genres, cast, keywords)
* Easy-to-use web interface built using Streamlit
* Pre-trained machine learning model and vectorizer for instant predictions
* Lightweight and deployable to platforms like Render or Streamlit Cloud

---

## 🎓 Concept

This project applies **content-based filtering** to recommend movies similar to a given one. The system uses textual metadata (overview, cast, genres, keywords) to compute similarity scores between movies.

### Types of Recommendation Techniques

* **Content-Based Filtering**: Recommends items with similar features.
* **Collaborative Filtering**: Based on the preferences of similar users (not used here).
* **Hybrid Methods**: Combine content and collaborative filtering.

Our project focuses on **content-based filtering**.

---

## 🧠 How It Works

1. **Data Preparation**

   * Combine movie metadata such as title, overview, cast, crew, genres, and keywords.

2. **Text Vectorization**

   * Convert combined metadata into numeric vectors using `CountVectorizer`.

3. **Similarity Calculation**

   * Use **cosine similarity** to measure how close each movie is to another in the vector space.

4. **Recommendation**

   * Select a movie → compute similarities → return top N similar movies.

All these are preprocessed and stored in:

* `vectorizer.pkl`: vectorizer object
* `model.pkl`: similarity matrix/model
* `features.pkl`, `bigmovies.pkl`: processed movie metadata

---

## 🛠 Installation

```bash
git clone https://github.com/Ashy-it24/Movie-Recommendation_ML.git
cd Movie-Recommendation_ML
pip install -r requirements.txt
```

---

## 💻 Usage

Launch the web application:

```bash
streamlit run app.py
```

Then open the local link (usually [http://localhost:8501](http://localhost:8501)) in your browser.

---

## 📂 Repository Structure

```
Movie-Recommendation_ML/
├── .idea/                   # Project settings (optional)
├── app.py                  # Main Streamlit web app
├── download.py             # Script to download movie data (optional)
├── vectorizer.pkl          # Pre-trained text vectorizer
├── model.pkl               # Pre-trained similarity model
├── features.pkl            # Feature set
├── bigmovies.pkl           # Cleaned movie metadata
├── movie_rec_ml.ipynb      # Jupyter notebook with model logic
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 📦 Dependencies

Key Python packages used:

* `pandas`
* `numpy`
* `scikit-learn`
* `streamlit`
* `nltk`

Install all using:

```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

Pull requests are welcome! Suggestions for improvements include:

* Adding collaborative filtering or hybrid model
* Improving UI design
* Expanding the metadata features (e.g., production company, language)

---

## 📄 License

This project is open-source and free to use for educational and non-commercial purposes.

---

Feel free to fork, star, and contribute to the repository! 🎥🍿
