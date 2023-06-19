from fastapi import FastAPI
import pandas as pd
from unidecode import unidecode
import re
import locale
locale.setlocale(locale.LC_TIME, 'es_ES') # setting a local configuration for dates values

df_movies = pd.read_csv("./processed_data/movies.csv")
df_cast = pd.read_csv("./processed_data/cast.csv")
df_crew = pd.read_csv("./processed_data/crew.csv")
actor_financial = pd.read_csv("./processed_data/actor_financial.csv")
director_financial = pd.read_csv("./processed_data/director_financial.csv")

df_movies["transformed_title"] = [x.lower().strip().replace(" ", "") for x in df_movies["title"]] # with pandas series, we need to use comprehension lists or apply()
df_movies["transformed_title"] = [unidecode(x) for x in df_movies["transformed_title"]]
df_movies["transformed_title"] = [re.sub(r'[^\w\s]', '', x) for x in df_movies["transformed_title"]]

actor_financial["transformed_name"] = [x.lower().strip().replace(" ", "") for x in actor_financial["name"]]  # with pandas series, we need to use comprehension lists or apply()
actor_financial["transformed_name"] = [unidecode(x) for x in actor_financial["transformed_name"]]
actor_financial["transformed_name"] = [re.sub(r'[^\w\s]', '', x) for x in actor_financial["transformed_name"]] 

director_financial["transformed_name"] = [x.lower().strip().replace(" ", "") for x in director_financial["name"]]  # with pandas series, we need to use comprehension lists or apply()
director_financial["transformed_name"] = [unidecode(x) for x in director_financial["transformed_name"]]
director_financial["transformed_name"] = [re.sub(r'[^\w\s]', '', x) for x in director_financial["transformed_name"]]  

df_movies["transformed_director"] = [x.lower().strip().replace(" ", "") for x in df_movies["director"]] # with pandas series, we need to use comprehension lists or apply()
df_movies["transformed_director"] = [unidecode(x) for x in df_movies["transformed_director"]]
df_movies["transformed_director"] = [re.sub(r'[^\w\s]', '', x) for x in df_movies["transformed_director"]]

app = FastAPI()

@app.get('/cantidad_filmaciones_mes/{mes}')
def cantidad_filmaciones_mes(mes: str):
    if type(mes) == str:
        mes = mes.lower().strip().replace(" ", "")
        mes = unidecode(mes)  # delete accents
        mes = re.sub(r'[^\w\s]', '', mes)  # delete special characters and punctuation marks
        # '%B' complete name of the month, strftime is string format time, it allows to format date data to a desirable representation
        # dt only works with pandas series that are datetime type
        df_movies["release_month"] = pd.to_datetime(df_movies["release_date"]).dt.strftime('%B')
        if mes in df_movies["release_month"].unique():
            count_by_month = df_movies.groupby(["release_month"])["title"].count()
            return {mes: count_by_month[mes].item()}  # needs item() because fastapi doesn't process numpy.int64 type objects
        else:
            return "Entered value is not valid."  # needs item() because fastapi doesn't process numpy.int64 type objects
    else:
        return "Entered value is not valid."
    
@app.get('/cantidad_filmaciones_dia/{dia}')
def cantidad_filmaciones_dia(dia: str):
    if type(dia) == str:
        dia = dia.lower().strip().replace(" ", "")
        dia = unidecode(dia) # delete accents
        dia = re.sub(r'[^\w\s]', '', dia) # delete special characters and punctuation marks
        # dt only works with pandas series that are datetime type
        # '%A' complete name of the day, strftime is string format time, it allows to format date data to a desirable representation
        df_movies["release_day"] = pd.to_datetime(df_movies["release_date"]).dt.strftime('%A') 
        if dia in df_movies["release_day"].unique():
            count_by_day = df_movies.groupby(["release_day"])["title"].count()
            return {dia:count_by_day[dia].item()}
        else:
            return "Entered value is not valid." 
    else:
        return "Entered value is not valid."
    