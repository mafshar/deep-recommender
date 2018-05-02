import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.recommendation._ // Uses ALS to determine latent factors in the matrix; Uses Rating class
import org.apache.spark.sql._ // Uses spark sql datatypes (needed for parsing)

// SQL Context (OPTIONAL: for data exploration purposes)
val sqlc = new org.apache.spark.sql.SQLContext(sc)
import sqlc.implicits._ // Uses sqlc datatypes

// Class Definitions:
// case class Movie(movieId: Int, title: String, genres: Seq[String])
case class Movie(movieId: Int, title: String) // drop the genres: not used in this recommender
case class User(userId: Int, gender: String)

// Helper functions (if using smaller dataset, delimeter would be "::"):
def isValidMovie(record: String) : Boolean = {
    val delimeter : String = ","
    val values = record.split(delimeter)
    values.length == 3
}

def createMovie(record: String) : Movie = {
    val delimeter : String = ","
    val values = record.split(delimeter)
    Movie(values(0).toInt, values(1).toString)
}

def isValidUser(record: String) : Boolean = {
    val delimeter : String = ","
    val values = record.split(delimeter)
    values.length == 4
}

def createUser(record: String) : User = {
    val delimeter : String = ","
    val values = record.split(delimeter)
    User(values(0).toInt, values(1).toString)
}

def isValidRating(record: String) : Boolean = {
    val delimeter : String = ","
    val values = record.split(delimeter)
    values.length == 4
}

def createRating(record: String) : Rating = {
    val delimeter : String = ","
    val values = record.split(delimeter)
    Rating(values(0).toInt, values(1).toInt, values(2).toDouble) // we ignore the timeStamp as that is not a useful feature
}

// files and parameters:
// val ratingsfile = "/user/ma2510/movies_data/ratings.dat"
// val moviesfile = "/user/ma2510/movies_data/movies.dat"

val ratingsfile = "/user/ma2510/movies_data_large/ratings.csv"
val moviesfile = "/user/ma2510/movies_data_large/movies.csv"

val modelfile = "/user/ma2510/recommender_models/recommendationModel"
val splitSeed = 0L

// data rdds
val rawRatings = sc.textFile(ratingsfile)
val filteredRatingsRec = rawRatings.first()
val ratings = rawRatings.filter(record => record != filteredRatingsRec).filter(isValidRating).map(createRating)

val rawMovies = sc.textFile(moviesfile)
val filteredMoviesRec = rawMovies.first()
val movies = rawMovies.filter(record => record != filteredMoviesRec).filter(isValidMovie).map(createMovie)

// train-val-test split
val splits = ratings.randomSplit(Array(0.8, 0.2), splitSeed)
val trainData = splits(0)
val testData = splits(1)

// for prediction
val testPredict = testData.map {
    case Rating(user, product, rate) => (user, product)
}

// Model parameters
val rank = 15
val iters = 20
val regularizationParameter = 0.1 // lambda
val blocks = -1 // autoconfigure parrellism
val modelSeed = 10L

// ALS.train(ratings, rank, iterations, lambda, blocks, seed)
val model = ALS.train(trainData, rank, iters, regularizationParameter, blocks, modelSeed)

val predictions = model.predict(testPredict).map {
    case Rating(user, product, rate) => ((user, product), rate)
}

val ratesAndPreds = testData.map {
    case Rating(user, product, rate) => ((user, product), rate)
}.join(predictions)

// @NOTE: r1: test rating, r2: predicted rating
// best mse: 0.6559886450815043
val mse = ratesAndPreds.map {
    case ((user, product), (r1, r2)) =>
    val error = (r1 - r2)
    error * error
}.mean()

// best mae: 0.6342475189415628
val mae = ratesAndPreds.map {
    case ((user, product), (r1, r2)) =>
    val error = (r1 - r2)
    Math.abs(error)
}.mean()

// save model to file for later predictions:
model.save(sc, modelfile)
