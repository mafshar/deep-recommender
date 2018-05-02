import org.apache.spark.mllib.recommendation._ // Uses ALS to determine latent factors in the matrix; Uses Rating class
import org.apache.spark.sql._ // Uses spark sql datatypes (needed for parsing)

// SQL Context
val sqlc = new org.apache.spark.sql.SQLContext(sc)
import sqlc.implicits._ // Uses sqlc datatypes

// Class Definitions:
// case class Movie(movieId: Int, title: String, genres: Seq[String])
case class Movie(movieId: Int, title: String) // drop the genres: not used in this recommender
case class User(userId: Int, gender: String, age: Int, occupation: Int, zipcode: String)

// Helper functions:
def isValidMovie(record: String) : Boolean = {
    val delimeter : String = "::"
    val values = record.split(delimeter)
    values.length == 3
}

def createMovie(record: String) : Movie = {
    val delimeter : String = "::"
    val values = record.split(delimeter)
    Movie(values(0).toInt, values(1).toString)
}

def isValidUser(record: String) : Boolean = {
    val delimeter : String = "::"
    val values = record.split(delimeter)
    values.length == 5
}

def createUser(record: String) : User = {
    val delimeter : String = "::"
    val values = record.split(delimeter)
    User(values(0).toInt, values(1).toString, values(2).toInt, values(3).toInt, values(4).toString)
}

def isValidRating(record: String) : Boolean = {
    val delimeter : String = "::"
    val values = record.split(delimeter)
    values.length == 4
}

def createRating(record: String) : Rating = {
    val delimeter : String = "::"
    val values = record.split(delimeter)
    Rating(values(0).toInt, values(1).toInt, values(2).toDouble) // we ignore the timeStamp as that is not a useful feature
}

// files and parameters:
val ratingsfile = "/user/ma2510/movies_data/ratings.dat"
val moviesfile = "/user/ma2510/movies_data/movies.dat"
val splitSeed = 0L

// data rdds
val ratings = sc.textFile(ratingsfile).filter(isValidRating).map(createRating)
val movies = sc.textFile(moviesfile).filter(isValidMovie).map(createMovie)

// train-val-test split
val splits = ratings.randomSplit(Array(0.8, 0.1, 0.1), splitSeed)
val train = split(0)
val validation = split(1)
val test = split(2)

// for prediction
// val validation_predict = validation.map {
//     // case Rating
// }

// Model parameters
val modelSeed = 10L
val rank = 10
val regularizationParameter = 0.1
val iters = 10
val error = 0
val tolerance = 0.02

val model = ALS.train(ratings, rank, iters, 0.01)

// getting the predictions y_hat
val predictions = model.predict(usersProducts)
    .map(record => Array((record(0), record(1)), record(2))) // ((user, product), rate))

// getting the ratings y and the predictions y_hat to calculate accuracy/precision/recall
val ratingsAndPred = ratings
    .map(record => Array((record(0), record(1)), record(2))) // ((user, product), rate))
    .join(predictions)

// NEXT STEPS:
// calculate the precision and recall
// use a metric to measure the understanding of the model and the accuracy/precision/recall
// explore methods in deep learning in order to make a deep collaborative filtering model (autoencoders)
