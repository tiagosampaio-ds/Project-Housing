# Project-Housing
## Source
https://www.kaggle.com/datasets/camnugent/california-housing-prices

## Context
This is the dataset used in the second chapter of Aurélien Géron's recent book 'Hands-On Machine learning with Scikit-Learn and TensorFlow'. It serves as an excellent introduction to implementing machine learning algorithms because it requires rudimentary data cleaning, has an easily understandable list of variables and sits at an optimal size between being to toyish and too cumbersome.

The data contains information from the 1990 California census. So although it may not help you with predicting current housing prices like the Zillow Zestimate dataset, it does provide an accessible introductory dataset for teaching people about the basics of machine learning.

## Content
The data pertains to the houses found in a given California district and some summary stats about them based on the 1990 census data. Be warned the data aren't cleaned so there are some preprocessing steps required! The columns are as follows, their names are pretty self explanitory:

1. longitude: A measure of how far west a house is; a higher value is farther west

2. latitude: A measure of how far north a house is; a higher value is farther north

3. housingMedianAge: Median age of a house within a block; a lower number is a newer building

4. totalRooms: Total number of rooms within a block

5. totalBedrooms: Total number of bedrooms within a block

6. population: Total number of people residing within a block

7. households: Total number of households, a group of people residing within a home unit, for a block

8. medianIncome: Median income for households within a block of houses (measured in tens of thousands of US Dollars)

9. medianHouseValue: Median house value for households within a block (measured in US Dollars)

10. oceanProximity: Location of the house w.r.t ocean/sea

## Objective

Answer two business questions

- as younger the housingMedianAge, higher the medianHousevalue?

- as higher the medianIncome, higher the medianHousevalue?

Get a better prediction than the book 'Hands-On Machine learning with Scikit-Learn and TensorFlow', the tuned model with the train set get a RMSE: 49.682 and the final evaluation was 47.730,2 for the test set. The strategy is try different methods os scaling, hyperparameters, algorithms and try eliminate some features

## Split train and test set
Before the split, we can see that is a high concentration of the maximum value on the target. The book told us that the value was capped to 500.001 when actually was beyond that. So we are going to exclude this values for the algorithm can learn that the price can go beyond that limit.

Following the example of the book tha we based on, the "median_income" has been told, by experts of the business,that it is the most important feature. To avoid a sampling bias in the test set over the data set, when we separate in train and test set, we create a new collumn to auxiliate that stratification with "median_income" and maintain the proportion.

## Feature engineering
The book suggest the creation of some features. Analysing the original features: rooms, bedrooms, population. Looking at them in a isolate form they seem to not tell us much, mainly because it´s not only the information of a house but the block. Combining this features to generate new one that explains more the context seems reasonable.

The features: rooms_per_household, bedrooms_per_room, population_per_household.

## Exploratory Data analysis
Univariate, bivariate and multivariate analysis were made.

Univariate analysis: With the help of histograms, it was seen that the features dont have a normal distribution, this is important for the pre processing part.

Bivariate analysis: With the help of a correlation heatmap we can confirm now that median income has a high correlation with the median house value which corroborates for the stratified split in the beginning. Other thing that we can notice it's a high correlation between the features: total rooms, total bedrooms anda households. Maybe this features are carrying the same information and its not necessary have them all. The regplot told us thatyounger the house, higher the median house value. However, higher the median income, higher the house value we can. The boxplot graphic show us that the ocean proximity has some influence on the median house value. When its inland, has the lowest median house value but when it is island, has the highest. This analysis is a litte damaged because only have 3 registers of the island variable.

 ## Pre processing
In this part, i created three pipelines with standard scaler, min max scaler and robust scaler. All of them use a simple imputer by median and before use the standard scaler, i transformed the data with the method yeo-johnson to be closer to a normal distribution. For the categorical features the method one hot encoder was used.
 
 ## Evaluation of machine learning models
The algorithms: Elastic net, Random forest, Stochastic Gradient Descent, Support vector regressor and Xgboost were trained using the model selection method cross validation. All of them were trained with different pipelines and with and without transforming the target feature to a normal distribution using a log function to transform and an exponential transform as inverse.

The best result was provide from XGBoost model with Standard scaler transformation and withou target transformation. 

RMSE = 43451.802251	
 
 ## Feature selection
Were plotted a learning curve that tells us that tThe training and cross validation score did not converge yet, so probably this model would benefit with more data. and a RFECV graphic that choose 14 features from 16, although the RFECV package has pointing 14 features as ideal, i optate to only exclude the population feature because the Island variable is associate with the ocean proximity feature. The main benefit would be make a easier deploy e more simple model.
 
 ## Hyperparameter fine tuning
I choose the method gridsearchCV to make a deep evaluation of the results with different combinations of hyperparameters.
Let's discuss about the hyperparameters that we will tune. 

First of all the xgboost has a lot of hyperparameters, i choose some of them that can adjust regularization better, to avoid overfitting and achive a good generalization. 

Eta: Step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative.

Subsample: Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. and this will prevent overfitting. Subsampling will occur once in every boosting iteration.

Max_depth: Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. 0 indicates no limit on depth. Beware that XGBoost aggressively consumes memory when training a deep tree. exact tree method requires non-zero value.

N_estimators: Number of gradient boosted trees. Equivalent to number of boosting rounds.

Random_state: state of randomness.

Source: https://xgboost.readthedocs.io/en/stable/parameter.html ; https://xgboost.readthedocs.io/en/stable/python/python_api.html ;
Book: "Machine learning: Guia de referência rápida" 

RMSE = 41495.87596092814

## Final model
In this part, the evaluation with the test set was made. 

RMSE = 39745.535561; MAE = 26640.835081 ; R² = 0.829436 

We reached a RMSE of 39.745 and the book had reached a RMSE of 47.730, so the model has been solidly improved.
