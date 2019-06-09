#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
dataset_movies = pd.read_csv('movies.csv')
dataset_credits = pd.read_csv('credits.csv')

#cleaning 
dataset_credits.rename(columns = {'movie_id': 'id'} , inplace = True)
dataset_credits = dataset_credits.drop(['title'] , axis =1)
dataset_credits = dataset_credits.merge(dataset_movies , on = 'id')

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words = 'english')
dataset_credits['overview'] = dataset_credits['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(dataset_credits['overview'])
tfidf_matrix.shape

from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix , tfidf_matrix)

indices= pd.Series(dataset_credits.index , index = dataset_credits['title'])

def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return dataset_credits['title'].iloc[movie_indices]

get_recommendations('The Dark Knight Rises')