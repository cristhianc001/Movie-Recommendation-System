# Content-Based Movie Recommendation System

![Word cloud for movie genres](https://raw.githubusercontent.com/cristhianc001/videogame-data-analysis/main/visualizations/gdp-population-scatterplot.png)  
*Word cloud for movie genres*


This is the implementation of Tf-idf (Term frequency – Inverse document frequency) in the development of a basic content-based movie recommendation system, deployed via FasApi and Render.

## Usage

The final result can be see in this Render [link](https://movie-recommendation-system-gbft.onrender.com/) but if the server is down, you can run it locally following these steps:

- Open your terminal and clone this repository in a directory of your liking with this code:<br>
`git clone https://github.com/cristhianc001/movie-recommendation-system`
- Install the requirements using `pip install -r requirements.txt`
- Execute the main file with `uvicorn main:app --reload`
- Open [localhost](http://localhost:8000/) in your browser or the adress that shows up in your terminal like this:
![Terminal after executing Uvicorn](https://raw.githubusercontent.com/cristhianc001/videogame-data-analysis/main/visualizations/gdp-population-scatterplot.png) 


## Project Structure

The project is organized as follows:

The repository is structured as follows:

- [`raw_data/`](raw_data/): Contains the pre-processed datasets in CSV format. The file size of the original datasets are high, so in case you want to execute the ETL yourself you should [download](https://drive.google.com/drive/folders/1RN0PqQ4cq9jMhDk1jx4S5OXc3Q5eHpco?usp=sharing) the files and put them inside the project folder. 
- [`processed_data/`](processed_data/): Contains the transformed and manipulated datasets in CSV format. Also includes extra files that helped in the building of the functions.
- [`notebooks/`](notebooks/): Includes Python notebooks for data cleaning, EDA, visualization and machine learning tasks. 
- [`img/`](img/): Includes Python figures for data visualization and images used around the repository.

## Contact

 Feel free to contact me here on Github or [LinkedIn](https://www.linkedin.com/in/cristhiancastro/) for any question about the project.
  