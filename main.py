from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from unidecode import unidecode
import re

## DATA LOADING AT THE BEGINNING SO THE FUNCTIONS DON'T HAVE TO DO IT EVERY TIME
df_movies = pd.read_csv("./processed_data/movies.csv")
df_cast = pd.read_csv("./processed_data/cast.csv")
df_crew = pd.read_csv("./processed_data/crew.csv")
actor_financial = pd.read_csv("./processed_data/actor_financial.csv")
director_financial = pd.read_csv("./processed_data/director_financial.csv")

app = FastAPI()
templates = Jinja2Templates(directory="./templates")

## ROOT
@app.get('/', response_class=HTMLResponse)
def welcome(request: Request):
    return templates.TemplateResponse("root.html", {"request": request})

## AMOUNT OF FILMS BY MONTH
@app.get('/cantidad_filmaciones_mes/{mes}')
def cantidad_filmaciones_mes(mes: str):
    if type(mes) == str:
        mes = mes.lower().strip().replace(" ", "")
        mes = unidecode(mes)  # delete accents
        mes = re.sub(r'[^\w\s]', '', mes)  # delete special characters and punctuation marks
        if mes in df_movies["release_month"].unique():
            count_by_month = df_movies.groupby(["release_month"])["title"].count()
            return {mes: count_by_month[mes].item()}  # needs item() because fastapi doesn't process numpy.int64 type objects
        else:
            return "Entered value is not valid."  # print doesn't work here
    else:
        return "Entered value is not valid."

## AMOUNT OF FILMS BY DAY    
@app.get('/cantidad_filmaciones_dia/{dia}')
def cantidad_filmaciones_dia(dia: str):
    if type(dia) == str:
        dia = dia.lower().strip().replace(" ", "")
        dia = unidecode(dia) # delete accents
        dia = re.sub(r'[^\w\s]', '', dia) # delete special characters and punctuation marks
        if dia in df_movies["release_day"].unique():
            count_by_day = df_movies.groupby(["release_day"])["title"].count()
            return {dia:count_by_day[dia].item()}
        else:
            return "Entered value is not valid." 
    else:
        return "Entered value is not valid."

## POPULARITY OF MOVIES  
@app.get('/score_titulo/{titulo}')
def score_titulo(titulo: str):
    if type(titulo) == str:
        titulo = titulo.lower().strip().replace(" ", "")
        titulo = unidecode(titulo)  # delete accents
        titulo = re.sub(r'[^\w\s]', '', titulo)  # delete special characters and punctuation marks      
        df_grouped = df_movies.groupby("transformed_title")["popularity"].sum()
        if titulo in df_grouped.index: # values of the grouped column are the new index in a grouped df
            normal_index = (df_movies["transformed_title"] == titulo).idxmax() # index for non transformed and non grouped values
            return {
                    "Title": df_movies["title"][normal_index], 
                    "Year": df_movies["release_year"][normal_index].item(), 
                    "Popularity" : df_grouped[titulo].round(2).item()
                    } 
        else:
            return "Entered value is not valid."
    else:
        return "Entered value is not valid."

## VOTES OF MOVIES    
@app.get('/votos_titulo/{titulo}')
def votos_titulo(titulo: str):
    if type(titulo) == str:
        titulo = titulo.lower().strip().replace(" ", "")
        titulo = unidecode(titulo)  # delete accents
        titulo = re.sub(r'[^\w\s]', '', titulo)  # delete special characters and punctuation marks
        df_grouped_total = df_movies.groupby("transformed_title")["vote_count"].sum()
        df_grouped_average = df_movies.groupby("transformed_title")["vote_average"].mean()
        if titulo in df_grouped_average.index: # values of the grouped column are the new index in a grouped df
             if df_grouped_total[titulo] >= 2000:
                normal_index = (df_movies["transformed_title"] == titulo).idxmax() # index for non transformed names and non grouped values
                return {
                        "Title": df_movies["title"][normal_index], 
                        "Year": df_movies["release_year"][normal_index].item(), # needs item() because fastapi doesn't process numpy.int64 type objects
                        "Total Votes" : df_grouped_total[titulo].item(), 
                        "Average Vote" :df_grouped_average[titulo].item()
                        }
             else:
                 return "Movie must have at least 2000 votes"
        else:
            return "Entered value is not valid."    
    else:
        return "Entered value is not valid."

## FINANCIAL DATA FROM ACTORS    
@app.get('/get_actor/{nombre_actor}')
def get_actor(nombre_actor):
    if type(nombre_actor) == str:
        nombre_actor = nombre_actor.lower().strip().replace(" ", "")
        nombre_actor = unidecode(nombre_actor)  # delete accents
        nombre_actor = re.sub(r'[^\w\s]', '', nombre_actor)  # delete special characters and punctuation marks
        if nombre_actor in actor_financial["transformed_name"].unique():
            index = (actor_financial["transformed_name"] == nombre_actor).idxmax()
            return {
                    'actor':actor_financial["name"][index], 'films':actor_financial["films"][index].item(), 
                    'total_return':actor_financial["total_return"][index].round(2).item(), 
                    'average_return':actor_financial["average_return"][index].item()
                    }        
        else:
            return "Entered value is not valid."
    return "Entered value is not valid."

## FINANCIAL DATA FROM DIRECTORS
@app.get('/get_director/{nombre_director}')
def get_director(nombre_director):
    if type(nombre_director) == str:
        nombre_director = nombre_director.lower().strip().replace(" ", "")
        nombre_director = unidecode(nombre_director)  # delete accents
        nombre_director = re.sub(r'[^\w\s]', '', nombre_director)  # delete special characters and punctuation marks
        if nombre_director in director_financial["transformed_name"].unique():
            index = (director_financial["transformed_name"] == nombre_director).idxmax()
            movies_director = df_movies[df_movies["transformed_director"] == nombre_director]
            movies_director = movies_director[["title","release_year", "return", "budget", "revenue"]]
            return {
                        'director':director_financial["name"][index], 
                        'total return':director_financial["total_return"][index].round(2).item(), 
                        'films': movies_director.to_dict(orient='records')
                     }
        else:
            return "Entered value is not valid."
    return "Entered value is not valid."
    