# Content-Based Movie Recommendation System

![Word cloud for movie genres](https://raw.githubusercontent.com/cristhianc001/movie-recommendation-system/main/img/wordcloud.png)  
*Word cloud for movie genres*


This is the implementation of [Tf-idf](https://es.wikipedia.org/wiki/Tf-idf) (Term frequency – Inverse document frequency) in the development of a basic content-based movie recommendation system, deployed via [FasApi](https://fastapi.tiangolo.com/) and [Render](https://render.com/).

The project consisted in a ETL phase where a dataset of movies, cast and directors had to be cleaned using Python libraries like Pandas, Numpy and AST, an API development stage with the building of seven functions and one of those functions gives a list of recommended movies supported by a similarity matrix and [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) to fit and transform the training data.

## Usage

The final result can be seen in this Render [link](https://movie-recommendation-system-gbft.onrender.com/) but if the server is down, you can run it locally following these steps:

- Open your terminal and clone this repository in a directory of your liking with this code:<br>
`git clone https://github.com/cristhianc001/movie-recommendation-system`
- Install the requirements using `pip install -r requirements.txt`
- Execute the main file with `uvicorn main:app --reload`
- Open [localhost](http://localhost:8000/) in your browser or the adress that shows up in your terminal like this:
![Terminal after executing Uvicorn](https://raw.githubusercontent.com/cristhianc001/movie-recommendation-system/main/img/uvicorn-screenshot.png) 


## Project Structure

The project is organized as follows:

The repository is structured as follows:

- [`raw_data/`](raw_data/): Contains the pre-processed datasets in CSV format. The file size of the original datasets are high, so in case you want to execute the ETL yourself you should [download](https://drive.google.com/drive/folders/1RN0PqQ4cq9jMhDk1jx4S5OXc3Q5eHpco?usp=sharing) the files and put them inside the project folder. 
- [`processed_data/`](processed_data/): Contains the transformed and manipulated datasets in CSV format. Also includes extra files that helped in the building of the functions.
- [`notebooks/`](notebooks/): Includes Python notebooks for data cleaning, EDA, visualization and machine learning tasks. 
- [`img/`](img/): Includes Python figures for data visualization and images used around the repository.

# TfidfVectorizer vs CountVectorizer

CountVectorizer is used to count the frequency of words in a corpus of documents. Creates a matrix where each row represents a document and each column represents a single word in the corpus. The values ​​in the matrix are the frequencies of the words in each document.

On the other hand, TfidfVectorizer is used to calculate the TF-IDF (Term Frequency-Inverse Document Frequency) value of words in a corpus. In addition to counting the frequency of words in each document, it also takes into account the rarity of words in the entire corpus. This means that words that are common to many documents get a lower TF-IDF value, while words that are more distinctive to a document get a higher TF-IDF value. <br>
Example:

    Document 1: El perro ladra.
    Document 2: El gato maulla.
    Document 3: El perro y el gato juegan.

CountVectorizer

                El   perro   ladra   gato   maulla   y   juegan
    Document 1   1     1       1       0      0      0     0
    Document 2   1     0       0       1      1      0     0
    Document 3   2     1       0       1      0      1     1


TfidfVectorizer

                    El   perro   ladra   gato   maulla   y juegan
    Document 1   0.33    0.47    0.47     0       0     0     0
    Document 2   0.33      0       0    0.47     0.47   0     0
    Document 3   0.47    0.33      0    0.33      0    0.33 0.33

# Cosine Similarity
Cosine similarity is a metric used to determine how similar the documents are irrespective of their size, measures the cosine of the angle between two vectors projected in a multi-dimensional space. In this context, the two vectors I am talking about are arrays containing the word counts of two documents.

$$\text{Cos}(\theta)=\frac{{\mathbf{A} \cdot \mathbf{B}}}{{\|\mathbf{A}\| \cdot \|\mathbf{B}\|}}$$

where A and B are the compared documents in form of vectors and θ is the angle between them. If the angle is zero, the result is 1, which means they are identical documents.<br>
Example:

```python
# Define the documents
doc_trump = "Mr. Trump became president after winning the political election. Though he lost the support of some republican friends, Trump is friends with President Putin"

doc_election = "President Trump says Putin had no political interference is the election outcome. He says it was a witchhunt by political parties. He claimed President Putin is a friend who had nothing to do with the election"

doc_putin = "Post elections, Vladimir Putin became President of Russia. President Putin had served as the Prime Minister earlier in his political career"

documents = [doc_trump, doc_election, doc_putin]
```
It's expected that cosine between documents number one and two, which are very similar, is closer to 1 that the cosine between doc two and three.


```python
# Scikit Learn
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Create the Document Term Matrix
count_vectorizer = CountVectorizer(stop_words='english')
count_vectorizer = CountVectorizer()
sparse_matrix = count_vectorizer.fit_transform(documents)
# OPTIONAL: Convert Sparse Matrix to Pandas Dataframe if you want to see the word frequencies.
doc_term_matrix = sparse_matrix.todense()
df = pd.DataFrame(doc_term_matrix, 
                  columns=count_vectorizer.get_feature_names(), 
                  index=['doc_trump', 'doc_election', 'doc_putin'])
```

As expected, the similarity between doc. one and two is higher than the other ones with a value of 0.48927489, meanwhile the cosine between one and three is 0.37139068 and between two and three is 0.38829014.

```python
# Compute Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity
print(cosine_similarity(df, df))
#> [[ 1.          0.48927489  0.37139068]
#>  [ 0.48927489  1.          0.38829014]
#>  [ 0.37139068  0.38829014  1.        ]]

```

# Conclusions and Recommendations
- There are movies with exact same name and its difficult to deal with them in some of the functions built here. One way to solve this is the concatenation between the title and the release year. Example: The Avengers (2012) and The Avengers (1998).
- A subset of the main file had to be used to create the feature and similarity matrices due to lack of computational resources. The amount of memory needed to support a matrix with the entire data was too high for the free plan of Render, so I decided to use a movies dataset filtered by vote count. The results could change if more resources are available.
- The convertion of the format of the datasets from CSV to another format like parquet could improve the performance of the deployment due to less memory consumption.
- LSA/LSI is very helpful to obtain better recommendations but needs extra computational resources. It can work with a small dataset, but the recommendations would be poor and the time of execution would be still very high. This an image from the Render console that shows how many minutes passed between the home and the recommendation function.
![Render Console](https://raw.githubusercontent.com/cristhianc001/movie-recommendation-system/main/img/time-model2.png)  
*Time of execution on Render console for LSA*
- The scope of this project only covers TF-IDF vectorizer, but there are other ways to develop a recommendation system, like the K-NN model used [here](https://www.analyticsvidhya.com/blog/2020/08/recommendation-system-k-nearest-neighbors/).

# Tech Summary
- [Pandas](https://pandas.pydata.org/docs/), [NumPy](https://numpy.org/doc/), [ast](https://docs.python.org/3/library/ast.html) are the libraries and modules used to clean the raw dataset.
- [Matplotlib](https://matplotlib.org/stable/index.html), [Seaborn](https://seaborn.pydata.org/), [WordCloud](https://pypi.org/project/wordcloud/) are the libraries used for visualization.
- [FastApi](https://fastapi.tiangolo.com/) is the library used to develop the API.
- [Scikit-Learn](https://scikit-learn.org/stable/) is the machine-learning library used to do vectorization and matrices calculation.
- [nltk](https://www.nltk.org/install.html) is the library used to test stemming and lemmatization.
- [Render](https://render.com/) is the service used to deploy the project.
- [Visual Studio Code](https://code.visualstudio.com/) is the code editor used for this project. Data Wrangler extension was also helpful.

## Extra Documentation
- [Cosine Similarity and TFIDF](https://medium.com/web-mining-is688-spring-2021/cosine-similarity-and-tfidf-c2a7079e13fa)
- [Finding Word Similarity using TF-IDF and Cosine in a Term-Context Matrix from Scratch in Python](https://towardsdatascience.com/finding-word-similarity-using-tf-idf-in-a-term-context-matrix-from-scratch-in-python-e423533a407)
- [Machine Learning 101: CountVectorizer Vs TFIDFVectorizer](https://enjoymachinelearning.com/blog/countvectorizer-vs-tfidfvectorizer/#:~:text=CountVectorizer%20simply%20counts%20the%20number,is%20to%20the%20whole%20corpus.)
- [Cosine Similarity – Understanding the math and how it works (with python codes)](https://www.machinelearningplus.com/nlp/cosine-similarity/)
- [Content-Based Movie Recommendation System Using BOW](https://www.youtube.com/watch?v=gtymDEKRr4A)
- [Step By Step Content-Based Recommendation System](https://medium.com/@prateekgaurav/step-by-step-content-based-recommendation-system-823bbfd0541c)
- [Latent Semantic Analysis: intuition, math, implementation](https://towardsdatascience.com/latent-semantic-analysis-intuition-math-implementation-a194aff870f8)
- [Stemming vs Lemmatization in NLP: Must-Know Differences](https://www.analyticsvidhya.com/blog/2022/06/stemming-vs-lemmatization-in-nlp-must-know-differences/#:~:text=Stemming%20is%20a%20process%20that,'%20would%20return%20'Car'.)

## Contact

 Feel free to contact me here on Github or [LinkedIn](https://www.linkedin.com/in/cristhiancastro/) for any question about the project.