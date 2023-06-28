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

## TRANSFORMING STRINGS TO LIST
df_train = df_movies[df_movies["vote_count"] >= 150].reset_index()
df_train["genres_list"] = [x if pd.isnull(x) else ast.literal_eval(x) for x in df_train["genres_list"]]
df_train["directors"] = [x if pd.isnull(x) else ast.literal_eval(x) for x in df_train["directors"]]
df_train["spoken_languages_list"] = [x if pd.isnull(x) else ast.literal_eval(x) for x in df_train["spoken_languages_list"]]
df_train["production_countries_list"] = [x if pd.isnull(x) else ast.literal_eval(x) for x in df_train["production_countries_list"]]
df_train["production_companies_list"] = [x if pd.isnull(x) else ast.literal_eval(x) for x in df_train["production_companies_list"]]

## EXTRACTING THE ELEMENTS OF THE LISTS
df_train["genres_list"] = [x if None else ", ".join(x) for x in df_train["genres_list"]]
df_train["directors"] = [x if None else ", ".join(x) for x in df_train["directors"]]
df_train["spoken_languages_list"] = [x if None else ", ".join(x) for x in df_train["spoken_languages_list"]]
df_train["production_countries_list"] = [x if None else ", ".join(x) for x in df_train["production_countries_list"]]
df_train["production_companies_list"] = [x if None else ", ".join(x) for x in df_train["production_companies_list"]]

## MAKING THE CORPUS OF THE TRAINING DATA
df_train["corpus"] = df_train["title"].fillna("") + ", " +df_train["genres_list"].fillna("") + ", " + df_train["overview"].fillna("") + ", " + df_train["directors"].fillna("") + ", " + df_train["collection"].fillna("") 

## TRANSFORMING TITLE TO EASE THE SEARCHING
df_train["transformed_title"] = [string_transformation(x) for x in df_train["title"]]

