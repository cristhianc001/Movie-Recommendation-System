import pandas as pd
import ast

## DATA LOADING AT THE BEGINNING SO THE FUNCTIONS DON'T HAVE TO DO IT EVERY TIME
df_movies = pd.read_csv("./processed_data/movies.csv")
df_crew = pd.read_csv("./processed_data/crew.csv")
actor_financial = pd.read_csv("./processed_data/actor_financial.csv")
director_financial = pd.read_csv("./processed_data/director_financial.csv")

df_train = df_movies[df_movies["vote_count"] >= 350].reset_index()
df_train["genres_list"] = [x if pd.isnull(x) else ast.literal_eval(x) for x in df_train["genres_list"]]
df_train["directors"] = [x if pd.isnull(x) else ast.literal_eval(x) for x in df_train["directors"]]
df_train["spoken_languages_list"] = [x if pd.isnull(x) else ast.literal_eval(x) for x in df_train["spoken_languages_list"]]
df_train["production_countries_list"] = [x if pd.isnull(x) else ast.literal_eval(x) for x in df_train["production_countries_list"]]
df_train["production_companies_list"] = [x if pd.isnull(x) else ast.literal_eval(x) for x in df_train["production_companies_list"]]

df_train["genres_list"] = [x if None else ", ".join(x) for x in df_train["genres_list"]]
df_train["directors"] = [x if None else ", ".join(x) for x in df_train["directors"]]
df_train["spoken_languages_list"] = [x if None else ", ".join(x) for x in df_train["spoken_languages_list"]]
df_train["production_countries_list"] = [x if None else ", ".join(x) for x in df_train["production_countries_list"]]
df_train["production_companies_list"] = [x if None else ", ".join(x) for x in df_train["production_companies_list"]]

df_train["corpus"] = df_train["title"].fillna("") + ", " +df_train["genres_list"].fillna("") + ", " + df_train["overview"].fillna("") + ", " + df_train["directors"].fillna("") + ", " + df_train["collection"].fillna("") 
