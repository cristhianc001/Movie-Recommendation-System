import pandas as pd
import ast
from unidecode import unidecode
import re

## AUX. FUNCTION TO TRANSFORM TITLES AND NAMES
def string_transformation(text):
    if type(text) == str:
        text = text.lower().strip().replace(" ", "")
        text = unidecode(text)  # delete accents
        text = re.sub(r'[^\w\s]', '', text)  # delete special characters and punctuation marks
        return text
    else:
     return "Entered value is not valid." 

## DATA LOADING AT THE BEGINNING SO THE FUNCTIONS DON'T HAVE TO DO IT EVERY TIME
df_movies = pd.read_csv("./processed_data/movies.csv")
df_crew = pd.read_csv("./processed_data/crew.csv")
actor_financial = pd.read_csv("./processed_data/actor_financial.csv")
director_financial = pd.read_csv("./processed_data/director_financial.csv")

## LOWERCASE THESE TO GET A MATCH IN THE QUERYS
df_movies["release_month"] = [x.lower() for x in df_movies["release_month"]]
df_movies["release_day"] = [x.lower() for x in df_movies["release_day"]]

## TRANSFORMING STRINGS TO LIST
df_movies["genres_list"] = [x if pd.isnull(x) else ast.literal_eval(x) for x in df_movies["genres_list"]]
df_movies["directors"] = [x if pd.isnull(x) else ast.literal_eval(x) for x in df_movies["directors"]]
df_movies["spoken_languages_list"] = [x if pd.isnull(x) else ast.literal_eval(x) for x in df_movies["spoken_languages_list"]]
df_movies["production_countries_list"] = [x if pd.isnull(x) else ast.literal_eval(x) for x in df_movies["production_countries_list"]]
df_movies["production_companies_list"] = [x if pd.isnull(x) else ast.literal_eval(x) for x in df_movies["production_companies_list"]]

## EXTRACTING THE ELEMENTS OF THE LISTS
df_movies["genres_list"] = [x if None else ", ".join(x) for x in df_movies["genres_list"]]
df_movies["directors"] = [x if None else ", ".join(x) for x in df_movies["directors"]]
df_movies["spoken_languages_list"] = [x if None else ", ".join(x) for x in df_movies["spoken_languages_list"]]
df_movies["production_countries_list"] = [x if None else ", ".join(x) for x in df_movies["production_countries_list"]]
df_movies["production_companies_list"] = [x if None else ", ".join(x) for x in df_movies["production_companies_list"]]


## MAKING THE CORPUS FOR MODEL
# genres and collection are added twice to give more weight to those attributes

df_movies["corpus"] = (df_movies["title"].fillna("") + ", " + df_movies["genres_list"].fillna("") 
                    + ", " + df_movies["overview"].fillna("") + ", " + df_movies["directors"].fillna("") + ", " + df_movies["collection"].fillna("") 
                    + ", " + df_movies["genres_list"].fillna("") + ", " + df_movies["collection"].fillna("") )
df_movies["corpus"]

## TRANSFORMING TITLE TO EASE THE SEARCHING
df_movies["transformed_title"] = [string_transformation(x) for x in df_movies["title"]]

## DATA FOR FIT
chosen_columns = ["title", "transformed_title", 'genres_list', "directors", "corpus"]
df_train = df_movies[df_movies["vote_count"] >= 250][chosen_columns].reset_index()


