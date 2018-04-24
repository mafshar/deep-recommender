# Movie Recommendation System

Mohammad Afshar, ma2510@nyu.edu

## Overview

### Background and Motivation
TBD

## Dataset

The data that will be used will be the MovieLens dataset for movie recommendations.

## Methodology
Recommendation system will be a collaborative filtering technique using the following two methodologies:

* Matrix factorization (using ALS)
* (Optional) Autoencoders

## Model
The model that will be used to come up with baseline recommendation predictions will be a collaborative filtering technique called Alternating Least Squares (ALS). The second model (optional) that might also be explored is a deep learning collaborative filtering technique involving autoencoders.

## Results
TBD

## Setup
Setup your data by following these instructions. The download link is [here](http://files.grouplens.org/datasets/movielens/ml-20m.zip)

* Locally, create a `/data` directory in the project root.

* Download all data files and expand them locally in `/data`.

When finished, your `/data` directory should look like this:

```
/data
  movies.csv
  ratings.csv
  tags.csv
```

To run the code:

```
spark-shell -i recommender.scala
```
