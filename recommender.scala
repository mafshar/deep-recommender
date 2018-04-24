// Uses ALS to determine latent factors in the matrix
// Use Rating class
import org.apache.spark.mllib.recommendation._

val datafile = "/users/ma2510/project/recommender/*"

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
