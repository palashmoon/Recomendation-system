#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
dataset_movie = pd.read_csv('movies.csv')
dataset_credits = pd.read_csv('credits.csv')

#merge two dataset
dataset_credits.rename(columns = {'movie_id': 'id'} , inplace = True)
dataset_movie = dataset_movie.merge(dataset_credits , on = 'id')

#to calculate IMDB score 
C = dataset_movie['vote_average'].mean()
m = dataset_movie['vote_count'].quantile(0.9)

#filter all the movies which has less vote_count than the min
q_movies = dataset_movie.copy().loc[dataset_movie['vote_count'] >= m]

#create a function for weighted score
def weighted_rating(x , m=m , C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m)*R) + (m/(v+m)*C)

q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
    
q_movies = q_movies.sort_values('score' , ascending = False)
q_movies[['title_x', 'vote_count' , 'cast' , 'score']].head(10)


pop= q_movies.sort_values('popularity', ascending=False)
plt.figure(figsize=(12,4))

plt.barh(pop['title_x'].head(6),pop['popularity'].head(6), align='center',
        color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Popular Movies")

