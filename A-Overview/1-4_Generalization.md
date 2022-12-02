# How 
### How they Generalize
Whether they work by simply comparing new data points to known data points, or instead by detecting patterns in the training data and building a predictive model, much like scientists do.

### #InstanceBasedLearning
The model learns the examples, then, when given a new case, it uses a similarity measure to find the most similar learned instances and uses them to make predictions.

### #ModelBasedLearning 
Tune some parameters to fit the model to the training set. Build a model and make predictions
*- Utility function: measures how good your model is*
*- Cost function: measures how bad it is, e.g., linear model to measure the distance between the predictions and the examples, try to minimize this distance*
*- Attribute = data type, features = predictors = attribute + its value*