import org.apache.spark.mllib.recommendation.ALS // Uses ALS to determine latent factors in the matrix
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating

val datafile = "/users/ma2510/project/recommender/*"

val data = sc.textFile(datafile)

val ratings = data
    .map(record => record.split(","))
    .map(record => Rating(user.toInt, item.toInt, rate.toDouble))

val usersProducts = ratings
    .map {
        case Rating(user, product, rate) => (user, product)
}

// Model parameters
val rank = 10
val iters = 10
val model = ALS.train(ratings, rank, iters, 0.01)
