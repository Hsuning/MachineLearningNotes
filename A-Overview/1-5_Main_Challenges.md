# Bad Data

## Insufficient Quantity of Training Data
It takes a lot of data for most ML algorithms to work properly. The unreasonable effectiveness of data - algo performed almost identically well once they were given enough data.

## Nonrepresentative Training Data
It is crucial to use a training set that is representative of the cases you want to generalize to.  
*Sampling noise* - the sample is too small.  
*Sampling bias* - the sampling method is flawed  
![[Pasted image 20220410200828.png]]

## Poor Quality Data
Model purpose is to detect the underlying patterns
- *outliers*: discard or fix the errors manually
- *missing:** ignore the attribute/instances, fill in, train 2 models - with and without it

# Irrelevant Feature
Garbage in, garbage out. Comping up with a good set of features
- *Feature selection* : most useful features
- *Feature extraction*: combining existing features to produce a more useful one
- *Creating new features* by gathering new data

# Overfitting the Training data
#Overfitting  
Overgeneralizing/ Overfitting means that the model performs well on the training data, but it does not generalize well. Especially for DNN that can detect subtle patterns in the data.
- *Simplify the model*: select a model with fewer parameters (liner model but not high-degree polynomial model), reduce the number of attributes, constrain the model (regularization).
- *More data*
- *Reduce the noise*: fix data errors and remove outliers.

>Degrees of freedom = how many parameters a model has. The linear model has 2 parameters, $\theta_1$ and $\theta_2$. If we forced $\theta_1=0$, then the model will only have 1 degree of freedom, so much harder time fitting the data properly.

# Hyperparameter
#Hyperparameter  
A parameter of a learning algorithm (not of the model)
- set before the learning process begins
- used to control the learning process
- tunable and can directly affect how well a model trains
- ex: topology and size of a neural network, regularization constant, number of clusters in a clustering algorithm
- the training algorithm learns the parameters from the data

## Underfitting the Training data
#Underfitting  
The model is too simple and cannot learn the underlying structure of the data. The predictions are inaccurate, even on the training examples.
- Select a more powerful model : with more parameters
- Feed better features
- Reduce the constraints on the model: reduce the regularization hyperparameter

## Generalization Error
#GeneralizationError  
Out-of-sample error, the error rate on new cases (test set), how well your model will perform on instances it has never seen before

## Hold-out Vvalidation dataset; development dataset
- hold out part of the training set to evaluate several candidate models and select the best one, to avoid adapting model and hyperparameters to produce the best model for test set.
- Cross-validation to evaluate once per validation set after it is trained on the rest of the data; averaging out the evaluations

## Data Mismatch
- A- Training set: pictures on webs
- B- Validation and test set: pictures collected with app, as representative as possible of the data you expect to use in production
- C- Train-dev set: hold out some of the training set  
If performances on A and C are good, but on B is bad = data mismatch; process A to make them look more like production data  
If performances on C is not good, overfitting A and need to simplify or regularize the model, clean up or enrich training data.

## No Free Lunch Theorem
- A model is a simplified version of the observations
- Discard the superfluous details that are unlikely to generalize to new instances
- Make some assumptions about the data, decide what data to keep, and evaluate only a few reasonable models
