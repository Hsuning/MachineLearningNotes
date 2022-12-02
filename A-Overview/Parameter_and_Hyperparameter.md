# Paramerter and Hyperparameter

```toc
```

## Parameter
#Parameter
- Also called as coefficients or weights
- the variables you can adjust during training, in order to improve the model
- A model has 1+ model parameters, they determine what it will predict given a new instance (e.g., the slope of a linear model). A learning algorithm tries to find optimal values for these parameters such that the model generalizes well to new instances.

## Hyperparameter
#Hyperparameter
- A hyperparameter is a parameter of the learning algorithm itself, not of the model (e.g., the amount of regularisation to apply).
- set before the learning process begins
- used to control the learning process
- tunable and can directly affect how well a model trains
- ex: topology and size of a neural network, regularization constant, number of clusters in a clustering algorithm
- the training algorithm learns the parameters from the data