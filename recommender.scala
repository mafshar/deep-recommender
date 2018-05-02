import org.apache.spark.mllib.recommendation._ // Uses ALS to determine latent factors in the matrix; Uses Rating class
import org.apache.spark.sql._ // Uses spark sql datatypes (needed for parsing)

// SQL Context
val sqlc = new org.apache.spark.sql.SQLContext(sc)
import sqlc.implicits._ // Uses sqlc datatypes

// Class Definitions:
case class Movie(movieId: Int, title: String, genres: Seq[String])
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
    Movie(values(0).toInt, values(1).toString, Seq(values(2)))
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

// datafiles:
val ratingsfile = "/user/ma2510/movies_data/ratings.dat"

val data = sc.textFile(datafile)

val ratings = data
    .map(record => record.split(","))
    .map(record => Rating(user.toInt, item.toInt, rate.toDouble))

val usersProducts = ratings
    .map(record => Array(user, product)

// Model parameters
val rank = 10
val iters = 10
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
