from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from dataframes import df_train, df_crew, actor_financial, director_financial, df_movies, string_transformation
from model import feat_matrix, get_recommendations

## INITIAL VALUES
app = FastAPI()
templates = Jinja2Templates(directory="./templates") # import use custom html templates
feature_matrix = None
tfidf_fit = None

### FUNCTIONS

## ROOT
@app.get('/', response_class=HTMLResponse) # output will be an HTML response
def welcome(request: Request):
    return templates.TemplateResponse("root.html", {"request": request}) # this represent a HTML request

## AMOUNT OF FILMS BY MONTH
@app.get('/cantidad_filmaciones_mes/{mes}')
def cantidad_filmaciones_mes(mes: str):
    month = string_transformation(mes)
    if month in df_movies["release_month"].unique():
        count_by_month = df_movies.groupby(["release_month"])["title"].count()
        return {month: count_by_month[month].item()} # needs item() because fastapi doesn't process numpy.int64 type objects 
    else:
        return "Entered value is not valid." # if we use print() instead of return, the output will be null

## AMOUNT OF FILMS BY DAY    
@app.get('/cantidad_filmaciones_dia/{dia}')
def cantidad_filmaciones_dia(dia: str):
    day = string_transformation(dia)
    if day in df_movies["release_day"].unique():
        count_by_day = df_movies.groupby(["release_day"])["title"].count()
        return {day:count_by_day[day].item()}
    else:
        return "Entered value is not valid." 

## POPULARITY OF MOVIES  
@app.get('/score_titulo/{titulo}')
def score_titulo(titulo: str):
    title = string_transformation(titulo)
    df_grouped = df_movies.groupby("transformed_title")["popularity"].sum()
    if title in df_grouped.index: # values of the grouped column are the new index in a grouped df
        normal_index = (df_movies["transformed_title"] == title).idxmax() # index for non transformed and non grouped values
        return {
                "Title": df_movies["title"][normal_index], 
                "Year": df_movies["release_year"][normal_index].item(), 
                "Popularity" : df_grouped[title].round(2).item()
                } 
    else:
        return "Entered value is not valid."

## VOTES OF MOVIES    
@app.get('/votos_titulo/{titulo}')
def votos_titulo(titulo: str):
    title = string_transformation(titulo)
    df_grouped_total = df_movies.groupby("transformed_title")["vote_count"].sum()
    df_grouped_average = df_movies.groupby("transformed_title")["vote_average"].mean()
    if title in df_grouped_average.index: # values of the grouped column are the new index in a grouped df
        if df_grouped_total[title] >= 2000:
            normal_index = (df_movies["transformed_title"] == title).idxmax() # index for non transformed and non grouped values
            return {
                    "Title": df_movies["title"][normal_index], 
                    "Year": df_movies["release_year"][normal_index].item(), 
                    "Total_Votes" : df_grouped_total[title].item(), 
                    "Average_Vote" : df_grouped_average[title].item()
                    }
        else:
            return "Movie must have at least 2000 votes"
    else:
        return "Entered value is not valid."

## FINANCIAL DATA FROM ACTORS    
@app.get('/get_actor/{nombre_actor}')
def get_actor(nombre_actor):
    name = string_transformation(nombre_actor)
    actor_financial["transformed_name"] = [string_transformation(x) for x in actor_financial["name"]]
    if name in actor_financial["transformed_name"].unique():
        index = (actor_financial["transformed_name"] == name).idxmax()
        return {
                'actor':actor_financial["name"][index], 
                'films':actor_financial["films"][index].item(), 
                'total_return':actor_financial["total_return"][index].round(2).item(), 
                'average_return':actor_financial["average_return"][index].item()
                }        
    else:
        return "Entered value is not valid."

## FINANCIAL DATA FROM DIRECTORS
@app.get('/get_director/{nombre_director}')
def get_director(nombre_director):
    name = string_transformation(nombre_director)
    director_financial["transformed_name"] = [string_transformation(x) for x in director_financial["name"]]
    if name in director_financial["transformed_name"].unique():
        index = (director_financial["transformed_name"] == name).idxmax() # obtain index corresponding the name
        df_director = df_crew[df_crew["job"] == "Director"] # filter jobs
        df_movies_director = df_director[["name", "id"]].merge(df_movies, how="left", on ="id") # join movies df with directors name
        df_movies_director.fillna("", inplace=True) # replace nan with blank spaces

        ## transforming the directors name so it can match with the input
        df_movies_director["transformed_director"] = [string_transformation(x) for x in df_movies_director["name"]] # with pandas series, we need to use comprehension lists or apply()

        ## filter the movies that matches the input name with the transformed name
        movies = df_movies_director[df_movies_director["transformed_director"] == name]
        movies = movies[["title","release_year", "return", "budget", "revenue"]]
        movies["release_year"] = [int(x) for x in movies["release_year"]] #change years like 1990.0 to 1990
        return {
                'director':director_financial["name"][index], 
                'total return':director_financial["total_return"][index].round(2).item(), 
                'films': movies.to_dict(orient='records') # orient=records doesn't show id and objects type
                }
    else:
        return "Entered value is not valid."
    
## MOVIE RECOMMENDATION

@app.get('/recomendacion/{titulo}')
def recommendations(titulo):
    global feature_matrix
    global tfidf_fit
    # setting this variables as global is useful because we only want to calculate this only one time
    # if this variables are calculated inside this function, they will replace the former values, in this case None
    if feature_matrix is None or tfidf_fit is None: 
        parameters = feat_matrix(df_train["corpus"])
        tfidf_fit = parameters[0]
        feature_matrix = parameters[1]

    recommendations = get_recommendations(titulo, tfidf_fit=tfidf_fit, feature_matrix=feature_matrix)
    return recommendations


    