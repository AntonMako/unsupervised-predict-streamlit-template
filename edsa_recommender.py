"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
from streamlit_option_menu import option_menu

# Data visualization
from collections import Counter
import re as regex
import plotly
from plotly import graph_objs
from time import time
import gensim
from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
from wordcloud import STOPWORDS
import matplotlib
import seaborn as sns 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import matplotlib.style as style 
sns.set(font_scale=1.5)
style.use('seaborn-pastel')
sns.set(style="whitegrid")
sns.set_style("dark")

# Data handling dependencies
import numpy as np 
import pandas as pd   
import re

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

movies = pd.read_csv('movies.csv')
tags = pd.read_csv('tags.csv')
imdb_data = pd.read_csv('imdb_data.csv')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# select the important features from imdb_data and merge with movies dataframe

df = imdb_data[['movieId','title_cast','director', 'plot_keywords']]
df = df.merge(movies[['movieId', 'genres', 'title']], on='movieId', how='inner')

# Before doing any string manipulation, one must convert records into strings

df['title_cast'] = df.title_cast.astype(str)
df['plot_keywords'] = df.plot_keywords.astype(str)
df['genres'] = df.genres.astype(str)
df['director'] = df.director.astype(str)

# Removing spaces in directors and actors names for countvector or tdifvectorizor optimatimization
# Removing spaces between names
df['director'] = df['director'].apply(lambda x: "".join(x.lower() for x in x.split()))
df['title_cast'] = df['title_cast'].apply(lambda x: "".join(x.lower() for x in x.split()))

# remove pipes from title_cast, Plot_keywords, and genres
# Getting the first five words in plot_keyword and first  3 actors names 
df['plot_keywords'] = df['plot_keywords'].map(lambda x: x.split('|')[:5])
df['plot_keywords'] = df['plot_keywords'].apply(lambda x: " ".join(x))
df['genres'] = df['genres'].map(lambda x: x.lower().split('|'))
df['genres'] = df['genres'].apply(lambda x: " ".join(x))
df['title_cast'] = df['title_cast'].map(lambda x: x.split('|')[:3])
df['title_cast'] = df['title_cast'].apply(lambda x: " ".join(x))

# new column with joined features to use in our cosine similarity for each movie
# Join title_cast, director, plot_keywords, and genres as a sentence
df['joined_features']= ''
joined_features=[]

# Columns to join

columns = ['title_cast', 'director', 'plot_keywords', 'genres']

# Combining the rows for the selected column for each movie

for i in range(0, len(df['movieId'])):
    words = ''
    for col in columns:
        words = words + df.iloc[i][col] + " "        
    joined_features.append(words)
    
# Add the joined information to the dataframe for recommendation system
df['joined_features'] = joined_features
df.set_index('movieId', inplace=True)
#for EDA
df1=df.copy()
#for Recommender system
df.drop(columns=['title_cast', 'director', 'plot_keywords', 'genres'], inplace=True)
df.to_csv('resources/data/recommend.csv', index=True)


# Data Loading
title_list = load_movie_titles('resources/data/recommend.csv')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","Movie Data", "EDA"]


    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[15200:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Movie Data":
        st.title("Exploring raw data")
        st.header("Movies, tags, imdb data, and user ratings")
        # st.subheader("Rating Count")
        # st.write("Users were very generous with their ratings")

        select_data=option_menu(
            menu_title="Select information",
            options=["Movies", "Tags","IMDB","User Rating", "Unique Features", "Data used to build recommnder app"],
            icons=["arrow-right-circle","arrow-right-circle","arrow-right-circle","arrow-right-circle"],
            menu_icon="hand-index-thumb-fill",
            default_index=0,        
        )

        if select_data == "Movies":
            st.write(movies.head(10))
            st.write("Shape of data")
            st.write(movies.shape)

        # Tags
        if select_data == "Tags":
            st.write(tags.head(10))
            st.write("Shape of data")
            st.write(tags.shape)

        #IMDB data

        if select_data == "IMDB":
            st.write(imdb_data.head(10))
            st.write("Shape of data")
            st.write(imdb_data.shape)

        # Ratings
        if select_data == "User Rating":
            st.write(train.head(10))
            st.write("Shape of data")
            st.write(train.shape)



        # Counting the unique features [movies, users, directors, actors, and user tags] as a pandas data frame.
        if select_data == "Unique Features": 
            features = pd.DataFrame({"number_movies": [len(movies['movieId'].unique().tolist())],
                       "movies_rated" : [len(train['movieId'].unique().tolist())],
                       "movies_tagged" : [len(tags['movieId'].unique().tolist())],
                       "number_users_rated" : [len(train['userId'].unique().tolist())],
                       "unique_user_tags": [len(tags['tag'].unique().tolist())],
                       "number_users_taggeds": [len(tags['userId'].unique().tolist())],
                       "number_directors" : [len(imdb_data['director'].unique().tolist())],
                       "number_actors" : [len(imdb_data['title_cast'].unique().tolist())]}, index=['unique_records'])
            st.write('number of unique records for features')
            st.write(features.transpose())

        #Data for recommender app

        if select_data == "Data used to build recommnder app":
            st.write(df.head(10))
            st.write("Shape of data")
            st.write(df.shape)

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.

    if page_selection == "EDA":
        st.title("Exploratory Data Analysis")
        st.header("Rating, Most Popular Movies, Movie count per year, movie genres, directors, actors")

        select_data=option_menu(
            menu_title="Select information",
            options=["Rating", "Most Popular Movies", "Movie count per year", "movie genres", "directors", "actors"],
            icons=["arrow-right-circle","arrow-right-circle","arrow-right-circle","arrow-right-circle"],
            menu_icon="hand-index-thumb-fill",
            default_index=0,        
        )

        # Rating

        if select_data == "Rating":
            ratings = train.merge(movies, on='movieId', how='inner')
            ratings.drop('timestamp', axis=1, inplace=True)
            fig=plt.figure(figsize=(15,10))
            sns.countplot(ratings.rating)
            st.pyplot(fig)

        # Most Popular movies
        if select_data == "Most Popular Movies":
            ratings = train.merge(movies, on='movieId', how='inner')
            ratings.drop('timestamp', axis=1, inplace=True)
            ratings_movie_mean = pd.DataFrame(ratings.groupby(['title'])[['rating']].mean())
            ratings_movie_mean['number_of_ratings'] = pd.DataFrame(ratings.groupby(['title'])['rating'].count())
            ratings_movie_mean['movie_popularity'] = pd.DataFrame(ratings.groupby(['title'])['rating'].count()*ratings.groupby(['title'])['rating'].mean())

            ratings_movie_mean.rename(columns = {'rating':'average_rating'}, inplace = True)
            ratings_movie_mean = ratings_movie_mean.sort_values(by='movie_popularity', ascending=False).head(10)
            st.write(ratings_movie_mean.head(10))

        # Movie Count per year

        if select_data == "Movie count per year":
            ratings = train.merge(movies, on='movieId', how='inner')
            ratings.drop('timestamp', axis=1, inplace=True)
            ratings['release_year'] = ratings.title.map(lambda x: re.findall('\((\d{4})\)', x))
            ratings.release_year = ratings.release_year.apply(lambda x: np.nan if not x else int(x[-1]))
            fig=plt.figure(figsize=(15,10))
            plt.ylabel('number of movies')
            sns.lineplot(data=ratings.groupby(['release_year'])['title'].count())
            st.pyplot(fig)


        # Movie genre

        if select_data == "movie genres":
            genres=[]
            for i in range(len(movies.genres)):
                for x in movies.genres[i].split('|'):
                    if x not in genres:
                        genres.append(x)  
            len(genres)
            for x in genres:
                movies[x] = 0
                for i in range(len(movies.genres)):
                    for x in movies.genres[i].split('|'):
                        movies[x][i]=1

            x={}
            for i in movies.columns[4:23]:
                x[i]=movies[i].value_counts()[1]
                st.write(print("{}    \t\t\t\t{}".format(i,x[i])))
            fig=plt.figure(4, figsize=(30,25))
            plt.ylabel('Count of genres', fontsize=20)
            plt.bar(height=x.values(),x=x.keys())
            plt.show()
            st.pyplot(fig)




        if select_data=="actors":
            text = " ".join(i for i in df1.title_cast)
            stopwords = set(STOPWORDS)
            more_stopwords = {'nan','seefullsummary'}
            stopwords = stopwords.union(more_stopwords)
            wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
            fig=plt.figure( figsize=(20,15))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.show()
            st.pyplot(fig)

        if select_data=="directors":
            text = " ".join(i for i in df1.director)
            stopwords = set(STOPWORDS)
            more_stopwords = {'nan','seefullsummary'}
            stopwords = stopwords.union(more_stopwords)
            wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
            fig=plt.figure( figsize=(20,15))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.show()
            st.pyplot(fig)










if __name__ == '__main__':
    main()
