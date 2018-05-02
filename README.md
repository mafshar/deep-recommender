# Movie Recommendation System

Mohammad Afshar, ma2510@nyu.edu

## Dataset

The data that will be used will be the MovieLens dataset for movie recommendations.

There were two distinct datasets used for the model trainging and prediction:
* [MovieLens 10 Million](http://files.grouplens.org/datasets/movielens/ml-10m.zip): 10 million ratings and 100,000 tag applications applied to 10,000 movies by 72,000 users
* [MovieLens 20 Million](http://files.grouplens.org/datasets/movielens/ml-20m.zip): 20 million ratings and 465,000 tag applications applied to 27,000 movies by 138,000 users

## Methodology
Recommendation system will be a collaborative filtering technique using the following two methodologies:

* Matrix factorization (using ALS)
* (Optional) Autoencoders

## Model
The model that will be used to come up with baseline recommendation predictions will be a collaborative filtering technique called Alternating Least Squares (ALS). The second model (optional) that might also be explored is a deep learning collaborative filtering technique involving autoencoders.

## Results
With an 80-20 split for the training and test sets, the parameters for the optimal model are as follows:
* rank: 15
* iterations: 20
* regularization parameter = 0.1

The results are:
* Large Dataset: MSE = 0.6452421631467037
* Large Datset: MAE = 0.62444593265524
* Small Dataset: MSE = 0.6659886450815043
* Small Datset: MAE = 0.6342475189415628

## Setup
Setup your data by following these instructions. Download from the above two links in "Data"

* Locally, create a `/data` directory in the project root.

* Download all data files and expand them locally in `/data`.

When finished, your `/data` directory should look like this:

```
/data
    /ml-10M100K
        movies.dat
        ratings.dat
        tags.dat
    /ml-20m
        movies.csv
        ratings.csv
        tags.csv
```

Setup on HDFS:

```
hdfs dfs -mkdir /$HOME/movies_data_large
hdfs dfs -mkdir /$HOME/recommender_models/recommendationModel
hdfs dfs -put movies.csv /$HOME/movies_data_large
hdfs dfs -put ratings.csv /$HOME/movies_data_large
hdfs dfs -put tags.csv /$HOME/movies_data_large
```

Build the model:
```
spark-shell -i recommender.scala
```
