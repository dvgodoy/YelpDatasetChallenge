# Yelp Dataset Challenge

## Summary

We developed classes to train two types of models using data of the [Yelp Dataset Challenge](https://www.yelp.com/dataset_challenge/). 

The first class aims at training recommender systems based on Apache Spark's ALS. It uses ratings data for a particular city and category of business to train the model. Predictions come in the form of the a top N list of recommended items for a given user.

You can find it at the [YelpRecommender](https://github.com/dvgodoy/YelpDatasetChallenge/blob/master/YelpRecommender.ipynb) notebook.

The second class aims at training LSTM neural networks with Keras, TensorFlow and GloVe embeddings. It uses review's text data and one its associated features (stars rating, usefulness, coolness and funiness) to train the model. Predictions are made for the feature the model was trained for.

You can find it at the [YelpLSTM](https://github.com/dvgodoy/YelpDatasetChallenge/blob/master/YelpLSTM.ipynb) notebook.

We made available Docker containers for:
- a recommender system for restaurants in Edinburgh;
- LSTM models for predicting stars rating, usefulness, coolness and funiness of a text review.

## Motivation

The Yelp Dataset contains a lot of review data: text, rating and stars. On one hand, we could benefit from experimenting with the text data and applying Deep Learning techniques to check what kind of features we would be able to predict from the text alone. On the other hand, we could try a more conservative approach and verify the quality of predictions we would be able to get from a model-based recommender.

We then decided to build both models. You will find out how to use them and how they were built in the following sections.

## How to use these models?

There is a specific Docker container for each model: DockerRecommender and DockerLSTM. Both containers will setup a tiny webserver based on CherryPy and Flask, providing endpoints to access each model's predictive capabilities:

#### Recommender:

    - port: 8001
    - method: GET
    - endpoints:
      - /random_user: returns the ID of a random user
      - /list?user=<user id>&n=<top n>: returns the N top ratings given by the user
      - /recommend?user=<user id>&n=<top n>: returns the top N recommendations to the user
      
Please refer to the [TestingRecommender](https://github.com/dvgodoy/YelpDatasetChallenge/blob/master/TestingRecommender.ipynb) notebook for an example of using the Recommender model.

You can also a Docker image directly:
```bash
docker run -p 8001:8001 -i -t dvgodoy/yelp-recommender:latest
```

#### LSTM:

    - port: 8002
    - method: POST
    - params:
      - review: text of the review
    - endpoints:
      - /stars: predicts the star rating (1 to 5) for the review
      - /useful: predicts if a review is useful (1) or not (0)
      - /cool: predicts if a review is cool (1) or not (0)
      - funny: predicts if a review is funny (1) or not (0)
      
Please refer to the [TestingLSTM](https://github.com/dvgodoy/YelpDatasetChallenge/blob/master/TestingLSTM.ipynb) notebook for an example of using the LSTM model.

You can also a Docker image directly:
```bash
docker run -p 8002:8002 -i -t dvgodoy/yelp-lstm:latest
```

## Exploratory Analysis

In this section, we  will explore the characteristics of the business and reviews files contained in the Yelp dataset.

The information comes from the following notebooks:

- [Businesses](https://github.com/dvgodoy/YelpDatasetChallenge/blob/master/ExploratoryBusinesses.ipynb)
- [Reviews](https://github.com/dvgodoy/YelpDatasetChallenge/blob/master/ExploratoryReviews.ipynb)

### Businesses

In the business file, we find 144.072 elements, each containing specific information about a venue, like its name, address, location, as well its corresponding attributes and categories as provided by Yelp's users.
First, we aggregate the business data in order to find the top cities, top categories and top attributes.

We can see the top cities concentrate the majority of the reviews, Las Vegas alone responding for 15.9% of all reviews.

Regarding the categories, the most frequently reviewed categories are Restaurants and Shopping.

With respect to the businesses' attributes, we find the acceptance of credit card, the price range and parking facilities to be the ones most frequent. It is also important to notice that the attributes contain a huge amount of missing data.

Top Cities | Top Categories | Top Attributes
:-:|:-:|:-:
![](/images/top_cities.png) | ![](/images/top10_categories.png) | ![](/images/top10_attributes.png)

Analysing the star ratings of the businesses, we can observe users have a positive bias, given the majority of businesses are rated 4 stars or more.
As the review counts go, we can observe a huge concentration of reviews on popular businesses. The coverage of the first 1/3 of businesses is as high as 80% of the total number of reviews (please notice the review figures were estimated on a downsampled dataset of reviews which is 10% the size of the original).

Businesses by Stars | Businesses by Review Count | Coverage of Reviews by Businesses
:-:|:-:|:-:
![](/images/dist_stars_business.png) | ![](/images/dist_review_business.png) | ![](/images/review_coverage_business.png)

### Reviews

For the reviews, we used a downsampled dataset which is 10% the size of the original size, approximately 410k reviews.

Once again, we can observe the positive bias of the user reviews, which are skewed towards the high end of the rating scale.

Regarding the coverage, we also observe a pattern similar to the one present in the businesses, although the concentration is not that strong, with approximately 1/4 of the users covering about 60% of the reviews.

Reviews by Stars | Coverage of Reviews by Users
:-:|:-:
![](/images/dist_stars_reviews.png) | ![](/images/review_coverage_user.png)

#### Sequences

Following the exploration of the downsampled reviews file, we focus our attention on the characteristics of the texts, namely, their sequence lengths.

Using a tokenizer, we find the dataset contains more than 48M tokens. Moreover, considering a dictionary of only 10k words, we are able to cover 97.32% of all tokens.

As the sequence lengths go, only a small subset of it has 400 or more tokens. But their distribution is quite distinct, when we consider the stars rating associated with it. In general, lower ratings are associated with longer sequences of tokens - one possible reason for this could be the fact that customers are more likely to write longer reviews to complain a bad service/product. We can also observe this in the scatterplot: even though the dispersion is significant, the trend is still noticeable.

Overall Sequence Lengths | Sequence Lengths by Stars | Sequence Lengths x Stars
:-:|:-:|:-:
![](/images/dist_seqlen_reviews.png) | ![](/images/dist_seqlen_reviews_stars.png) | ![](/images/scatter_seqlen_stars.png)

Then, we analysed the relationship between the sequence lengths and different features associated with the reviews: usefulness, coolness and funiness. Considering any of the features, users tend to consider more longer reviews: sequence lengths of useful, funny or cool reviews are longer than their counterparts.

Sequence Lengths by Useful | Sequence Lengths by Cool | Sequence Lengths by Funny
:-:|:-:|:-:
![](/images/dist_seqlen_reviews_useful.png) | ![](/images/dist_seqlen_reviews_cool.png) | ![](/images/dist_seqlen_reviews_funny.png)

#### Features

Now, we analyse the distribution of reviews, given the number of users who considered them either useful, cool or funny. In all three cases, the proportion of reviews with more than 2 users considering them useful, cool or funny is, respectively, 11.32%, 4.77% and 4.03%.

Reviews by Useful | Reviews by Cool | Reviews  by Funny
:-:|:-:|:-:
![](/images/dist_reviews_useful.png) | ![](/images/dist_reviews_cool.png) | ![](/images/dist_reviews_funny.png)

Next, we decided to analyse how this three features are related to the stars rating given by the user who wrote the review. It seems the more useful reviews are the ones with lower ratings - users may find a negative review particulary useful to avoid getting a bad deal. On the other hand, negative reviews are also considered funnier! And as cool reviews go, users seem to find higher ratings cooler - about 31% of 4-star reviews and 25% of 5-star reviews are considered cool.

Features by Stars | Table
:-:|:-:
![](/images/proportion_features_stars.png) | ![](/images/proportion_features_stars_table.png)

Moreover, we analysed what is the average sequence length, given the star rating and the three review features. We find the longer reviews are more likely considered useful, cool AND funny, all at once. Again, we can see that 1 and 2-star ratings are longer. It also seems to be the case useful reviews are longe than funny ones, which are longer than cool ones.

![](/images/seqlen_avg_features.png)

# Models

We trained two models based on the data - one recommender system for restaurants in Edinburgh and one LSTM neural network to predict both the stars rating given the text of a review, and its different features: usefulness, coolness and funiness.

## Recommender

For the recommender system, we used a model-based approach given by the Alternate Least Squares (ALS) algorithm implemented in Apache Spark.

To train and test this model, we took a subset of restaurants in Edinburgh - the European city with the biggest number of reviews in the dataset. Given the very high sparsity of the data (more than 99%), we reduced it by considering only users who have written more than 4 reviews. We also centered each user's mean rating.

To perform the split between training and test datasets, we took the last review written by a user as part of the test set and all the remaining ones as part of the training set.

Moreover, doing a random split would bring problems with the RMSE evaluator in Spark, as it always produce NaN whenever a given user is present on the test set only (even though there is a pull request for solving this issue already).

We developed a class to help organize the procedure of training a recommender system and perform a grid search over three available parameters in Spark's ALS: regularization parameter, rank (# of latent factors) and number of iterations.

The best model, chosen with a regularization parameter of 0.3, 10 latent factors and 5 iterations, performed with a training RMSE of 0.7234 and a test RMSE of 0.9429.

The figure below shows the scatter plot of actual and predicted ratings in the test set, indicating a somewhat poor performance of the model.

Future works could include training a similar model based on a bigger amount of data (using restaurants in Las Vegas, for instance), which would also allow to a more proper split of the data into training, validation and test sets. We could also experiment with the sparsity of the model, considering different thresholds for the number of reviews per user.

![](/images/recommender_actual_predicted.png)

## LSTM

For the LSTM neural network, we used Keras with the TensorFlow backend and GloVe embeddings (from Stanford NLP group - http://nlp.stanford.edu/projects/glove/).

Given the characteristics of the reviews presented in previous sections, we chose to a vocabulary of 10k words and a maximum sequence length of 400, which provide coverage for 95% of more of the cases.

Moreover, given the user's bias towards higher ratings, we balanced the dataset for each different model by downsampling the majority class(es), which resulted in smaller training sets for some models.

For the embeddings, we chose to use the 100-dimension vectors, as bigger vectors would increase the cost of training the models.

For the structure of the neural network itself, we experimented with a couple of different designs:
- LSTM with 1 or 2 layers
- GRU with 1 or 2 layers

We decided to train the models using 1-layer LSTM networks, as they seemed to provide somewhat more stable learning curves than its GRU counterparts, and also a quite similar level of accuracy of its 2-layer counterpart, which takes a lot more time to be trained.

Regarding the number of memory neurons, we experimented with 50 and 100 neurons, keeping the latter configuration as it produced better levels of accuracy. Again, bigger models were not experimented with due to computational power constraints.

We developed a class to help organize the procedure of training the LSTM neural networks, allowing to define most of the parameters: target feature, number of memory neurons, etc. The architecture of the network is not subject to configuration, though.

The hardware used was a GTX 1060, and each epoch took between 4 and 10 minutes, depending on the model and the size of the training set.

We trained 4 different classification models, one for each feature we are trying to predict: stars, usefulness, coolness and funiness. We used the early stopping callback with a patience of 2 epochs, meaning the model would stop training should the level accuracy fail to improve for 3 epochs in a row.

The results are not particularly striking, with accuracy levels in the order of 60-70% and ROC curves where 80% TPR come at a cost of 40% FPR. Besides, most models would flatten or worsen its accuracy curves by the 4th/5th epoch, suggesting overfitting. But they are nonetheless interesting, considering the simplicity of the network architecture and the high level of dataset downsampling involved in this exercise.

Future works could include deeper/different architectures and more expensive training on the entire dataset.

### Stars

For the stars rating prediction, the following model was trained:

![](/images/model_stars_lstm_100n.png)

Now we present the accuracy and loss curves. We can observe the validation accuracy does not improve much after the 5th epoch, staying below the level of 60%. 

![](/images/acc_loss_stars_lstm_100n.png)

Next, we present different metrics for classification of the reviews into different star ratings, as well as the confusion matrix.

We can see that the model performs better at both extremes of the rating scale: 1 and 5-star ratings. Even though the performance for toher classes, it is still worth noticing the model is not likely to confuse 1 and 5-star ratings.

![](/images/prfs_stars_lstm_100n.png)
![](/images/cm_stars_lstm_100n.png)

### Useful

For the usefulness prediction, the following model was trained:

![](/images/model_useful_lstm_100n.png)

Differently than the stars rating, the usefulness of a review is given by the number of users who considered that particular review useful - which has no upper bound by definition. Therefore, we could choose to consider a review as useful if only a single user had considered it so, or using a different threshold, say, 3 users.

We then trained an experimental model, using only reviews which were not considered useful at all (0) or were considered useful for at most 1 user. The model was not able to distinguish between the two classes, indicating that "slightly useful" reviews may not be significantly different than "not useful" reviews.

Nonetheless, we chose to train two models, one for each threshold - 1+ and 3+ clicks - and compare the performance of them.

Now we present the obtained accuracy and loss curves, metrics, confusion matrices and ROC curves.

We can see that the higher threshold produces significantly better results: accuracy improves from 64% to 69%, with better both precision and recall for the positive cases and an AUC improvement from 0.68 to 0.76. We can also observe the different confusion matrices: in the lower threshold, useful reviews are mistakenly predicted as not useful. Given the results of our experimental model, it is likely the case the model is considering the "slightly useful" reviews (which have only 1 or 2 clicks) as not useful, therefore hurting the metrics and the confusion matrix.

| Threshold - 1 or more useful clicks | Threshold - 3 or more useful clicks |
|:-:|:-:|
|![](/images/acc_loss_useful_lstm_100n_thresh0.png) | ![](/images/acc_loss_useful_lstm_100n_thresh2.png)|
|![](/images/prfs_useful_lstm_100n_thresh0.png) | ![](/images/prfs_useful_lstm_100n_thresh2.png)|
|![](/images/cm_useful_lstm_100n_thresh0.png) | ![](/images/cm_useful_lstm_100n_thresh2.png)|
|![](/images/roc_useful_lstm_100n_thresh0.png) | ![](/images/roc_useful_lstm_100n_thresh2.png)|

### Cool

For the coolness prediction, the following model was trained:

![](/images/model_cool_lstm_100n.png)

As with the usefulness of the review, we trained models for coolness using both thresholds.

Now we present the obtained accuracy and loss curves, metrics, confusion matrices and ROC curves.

Again, we can see that the higher threshold produces significantly better results: accuracy improves from 64% to 72%, with better both precision and recall for both positive and negative cases and an AUC improvement from 0.67 to 0.79. We observe the same phenomena as before in the two confusion matrices: in the lower threshold, cool reviews are mistakenly predicted as not cool, suggesting the use of the higher threshold to improve the results of the model.

| Threshold - 1 or more useful clicks | Threshold - 3 or more useful clicks |
|:-:|:-:|
|![](/images/acc_loss_cool_lstm_100n_thresh0.png) | ![](/images/acc_loss_cool_lstm_100n_thresh2.png)|
|![](/images/prfs_cool_lstm_100n_thresh0.png) | ![](/images/prfs_cool_lstm_100n_thresh2.png)|
|![](/images/cm_cool_lstm_100n_thresh0.png) | ![](/images/cm_cool_lstm_100n_thresh2.png)|
|![](/images/roc_cool_lstm_100n_thresh0.png) | ![](/images/roc_cool_lstm_100n_thresh2.png)|

### Funny

For the funiness prediction, the following model was trained:

![](/images/model_funny_lstm_100n.png)

As with the two previous models, we trained models for funiness using both thresholds.

Now we present the obtained accuracy and loss curves, metrics, confusion matrices and ROC curves.

One more time, we can see that the higher threshold produces significantly better results: accuracy improves from 64% to 72%, with better both precision and recall for both positive and negative cases and an AUC improvement from 0.69 to 0.79. BUT, for this particular model, confusion matrices behave differently: even though there is an improvement by chosing a higher threshold, the model still predicts reviews which are not actually funny, as funny.

| Threshold - 1 or more useful clicks | Threshold - 3 or more useful clicks |
|:-:|:-:|
|![](/images/acc_loss_funny_lstm_100n_thresh0.png) | ![](/images/acc_loss_funny_lstm_100n_thresh2.png)|
|![](/images/prfs_funny_lstm_100n_thresh0.png) | ![](/images/prfs_funny_lstm_100n_thresh2.png)|
|![](/images/cm_funny_lstm_100n_thresh0.png) | ![](/images/cm_funny_lstm_100n_thresh2.png)|
|![](/images/roc_funny_lstm_100n_thresh0.png) | ![](/images/roc_funny_lstm_100n_thresh2.png)|
