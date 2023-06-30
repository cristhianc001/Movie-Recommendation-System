from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
from dataframes import df_train, df_movies, string_transformation

vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")

def feat_matrix(corpus):
    tfidf_fit = vectorizer.fit(corpus)
    feature_matrix = tfidf_fit.transform(corpus)
    return tfidf_fit, feature_matrix # tuple of fitting data and feature matrix

def get_recommendations(title, tfidf_fit, feature_matrix):
    
    title = string_transformation(title)
    if title in df_movies["transformed_title"].unique():
        # vectorization of the corpus of entered movie   
        new_movie_vector = tfidf_fit.transform(df_movies[df_movies["transformed_title"] == title]["corpus"])

        # similarity matrix between movies in df_train and entered movie
        similarity_matrix = cosine_similarity(new_movie_vector, feature_matrix)

        # the product of the feature matrix and the entered movie vector is a matrix where [0] is the
        # similarity between that movie and the movies in df_train
        # output of enumarate is a tuple (movie index, score) 
        similar_movies = list(enumerate(similarity_matrix[0]))

        # if the movie is in the df_train, ignores the first recommendation because it's the same introduced movied.
        # that means that the slicing of the sorted movies should start at postion 1.
        if title in df_train["transformed_title"].unique():

            # the lambda function sort the list of tuples by the second position x[1]
            sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:6]

            recommendations = []
            for index, score in sorted_similar_movies: # iter the list of tuples and store values
                movie_title = df_train['title'][index]
                genres = df_train['genres_list'][index]
                director = df_train['directors'][index]
                similarity_score = score.round(4)

                recommendation = {
                    'title': movie_title,
                    'genres': genres,
                    'director': director,
                    'similarity': similarity_score
                }
                recommendations.append(recommendation)

            return recommendations
        
        else: # if the movie is NOT in the df_train, the slicing of similar movies list will start at postion 0
             sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[0:5]

             recommendations = []
             for index, score in sorted_similar_movies: # iter the list of tuples and store values
                movie_title = df_train['title'][index]
                genres = df_train['genres_list'][index]
                director = df_train['directors'][index]
                similarity_score = score.round(4)

                recommendation = {
                    'title': movie_title,
                    'genres': genres,
                    'director': director,
                    'similarity': similarity_score
                }
                recommendations.append(recommendation)

             return recommendations           
    else:
        return "Entered value is not valid."