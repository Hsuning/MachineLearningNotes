# Definition of Machine Learning

## Table of Contents
1. [[#Definition|Definition]]
2. [[#To Solve 4 Types of Problems|To Solve 4 Types of Problems]]
3. [[#Parameter|Parameter]]
4. [[#Hyperparameter|Hyperparameter]]

## Definition
ML is about giving computers the ability to get better at some task by learning from data, instead of having to explicitly code rules.

A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience.

T = flag spam fro new emails  
E = training data  
P = need to be defined, e.g., ratio of correctly classified emails

## To Solve 4 Types of Problems
- complex problems - no algorithmic solution
- replace long lists of hand-tuned rules
- build systems that adapt to fluctuating environments
- help humans learn (e.g., data mining).

## Parameter
#Parameter
- Also called as coefficients or weights
- the variables you can adjust during training, in order to improve the model
- A model has 1+ model parameters, they determine what it will predict given a new instance (e.g., the slope of a linear model). A learning algorithm tries to find optimal values for these parameters such that the model generalizes well to new instances.

## Hyperparameter
#Hyperparameter
- A hyperparameter is a parameter of the learning algorithm itself, not of the model (e.g., the amount of regularization to apply).
