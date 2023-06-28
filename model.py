from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
from dataframes import df_train, string_transformation

vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")

def feat_matrix(corpus):
    feature_matrix = vectorizer.fit_transform(corpus)
    return feature_matrix

def sim_matrix(feature_matrix):
    similarity_matrix = cosine_similarity(feature_matrix)
    return similarity_matrix

def get_recommendations(title, feature_matrix, similarity_matrix):
    # index of the entered title
    title = string_transformation(title)
    if title in df_train["transformed_title"].unique():
        movie_index = (df_train['transformed_title'] == title).idxmax()

        # get the similarity scores of the entered movie
        similarity_scores = similarity_matrix[movie_index]

        # obtain the index of the top similar movies
        top_indices = similarity_scores.argsort()[::-1][1:6]

        # create a list of dictionaries with the recommendations
        recommendations = []
        
        for index in top_indices:
            movie_title = df_train['title'][index]
            genres = df_train['genres_list'][index]
            director = df_train['directors'][index]
            similarity_score = similarity_scores[index].round(4)
            
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