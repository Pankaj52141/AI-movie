import gradio as gr
import pandas as pd
from surprise import SVD, Dataset, Reader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
movies = pd.read_csv("data/ml-latest-small/movies.csv")
ratings = pd.read_csv("data/ml-latest-small/ratings.csv")

# Content-based filtering
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_content(title):
    idx = movies[movies['title'].str.contains(title, case=False)].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# Collaborative filtering
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings[['userId','movieId','rating']], reader)
trainset = data.build_full_trainset()
svd = SVD()
svd.fit(trainset)

def hybrid_recommendation(user_id, title):
    content_recs = recommend_content(title)
    predictions = [(t, svd.predict(user_id, movies[movies['title']==t]['movieId'].values[0]).est) for t in content_recs]
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    return [t for t, _ in predictions]

# Gradio UI
def recommend_ui(user_id, movie_title):
    return hybrid_recommendation(user_id, movie_title)

iface = gr.Interface(
    fn=recommend_ui,
    inputs=[gr.Number(label="User ID"), gr.Textbox(label="Movie Title")],
    outputs=gr.Textbox(label="Recommended Movies"),
    title="Movie Recommendation System AI",
    description="Enter a movie you like and your User ID to get AI recommendations!"
)

iface.launch()
