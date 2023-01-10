# 2_Evaluation & Diagnostics
```toc
```

**How quickly you can get a machine learning system to work well will depend on large part on how well you can repeatedly make good decisions about what to do next**

## Machine Learning Diagnostic
- Diagnostic: a test that we run to
	- gain insight into what is/isn't working with a learning algorithm
	- gain guidance into improving its performance
- Diagnostics can take time to implement but doing so can be a very good use of time

## Evaluating models
#Overfitting #GeneralisationError
> Model fits the training data well but will fail to generalise to new examples not in the training set.

- Plotting the model (f(x)) might be a good way to see the curve, however, it is difficult to plot a model with more than 2 variables

### Step 1 - Split Dataset into **Training, Cross-validation and Test set**
#TrainSet #ValidationSet #TestSet
- Creating three data sets allows you to
	- train your parameters $W,B$ with the training set
	- tune model parameters such as complexity, regularisation and number of examples with the cross-validation set
- Evaluate your 'real world' performance using the training vs cross-validation performance provides insight into a model's propensity towards overfitting (high variance) or underfitting (high bias)

| data                                               | % of total | Description                                                                                                                                                                                  |
| -------------------------------------------------- |:----------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| training                                           |     60     | To fit the parameters of the model. Data used to tune model parameters $w$ and $b$ in training or fitting                                                                                    |
| cross-validation (validation set, development set) |     20     | To check the validity or really the accuracy of different model. Data used to tune other model parameters like degree of polynomial, regularization or the architecture of a neural network. |
| test                                               |     20     | To evaluate the model on new data. Data used to test the model after tuning to gauge performance on new data                                                                                 |
|                                                    |            |                                                                                                                                                                                              |

### Step 2 - Fit Parameters Using Training Set

### Step 3 - Systematically Evaluate How Well Our Learning Algorithm is Doing
By Computing Both $J_{test}, J_{cv}, J_{train}$ (train, Cross-validation and Test error). Ex: for #RegularizedLinearRegression  
![](Pasted%20image%2020230106151931.png)

> If overfitting, the cost for $J_{train}$ will be low and $J_{test}$ will be high.  
> (Not the best procedures) Picking the model with $J_{test}$ is likely to be an optimistic estimate of generalisation error, lower than actual estimate.

### Step 4 - Make Decision without Test Set
- Measure how the model is doing on the cross-validation and training set
- Ensure we haven't accidentally fit anything to the test set, so we can get a fair but not overly optimistic estimate of the generalisation error
- Pick the model with cross validation set, the one with lowest cv error

### Step 5 - Estimate Generalisation Error with Test Set
- $J_{test}$: is better estimate of how well the model will generalise to new data
- For classification, if we want to get the fraction of the test set and the fraction of the train set that the algorithm has misclassified, we can simply count $\hat{y} \neq y$ => $J_{test}, J_{cv}, J_{train}$ will be the fraction of the test set that has been misclassified

### Example
Choosing a neural network architecture for handwritten digit classification
- We want to know how many layer ? how many hidden units per layer ?
- So build several models
- Then evaluate the performance using cv by computing cost as fraction of cv examples that the algorithm has misclassified
- Then pick the model with the lowest cv error
- Then estimate the generalisation error with test set

## Error analysis
#ErrorAnalysis
- Manually examine misclassified examples and categorize them based on common traits
	- ex: spam mail classification
	- Deliberate misspellings (w4tches, med1cine)
	- Unusual email routing
	- Spam message in embedded image

## Overfitting (Variance) and Underfitting (Bias)
#Overfitting #Underfitting

### Definition
Data
- Extreme examples (outliers) can increase overfitting
- #NorminalVariable examples can reduce overfitting

![[Pasted image 20221130101833.png]]

| Underfit                                                                                                      | Just Right                          | Overfit                                                                 |
| ------------------------------------------------------------------------------------------------------------- | ----------------------------------- | ----------------------------------------------------------------------- |
| Does not fit the training set well                                                                            | Fits training set pretty well       | Fits the training set extremely well                                    |
| High bias                                                                                                     | generalization                      | High variance                                                           |
| There is a clear pattern shows that algorithm is unable to capture the data or with very strong preconception | Make good predictions with new data | End up with totally different predictions or highly variant predictions |
| $J_{train}, J_{cv}$ are high                                                                                                              | $J_{train}, J_{cv}$ are low                                    | $J_{train}$ is low but $J_{cv}$ is high                                                                        |

**The temperature is just right to drink.**

### Diagnosing Bias and Variance
![](Pasted%20image%2020230109103913.png)

- For simple linear regression, the training error will tend to go down when fitting a higher and higher order polynomial
- When the degree of polynomial is low, the $J_{cv}$ is high (underfitting); When the degree is high, the $J_{cv}$ is high too (overfitting); if the degree is right, $J_{cv}$ is much better
- High bias = #Underfitting = left most portion of the curve
	- $J_{train}$ will be high
	- $J_{train} \approx J_{cv}$: train and cv closer to each other
- High variance = #Overfitting = right most portion of the curve
	- $J_{train}$ my be low
	- $J_{cv} \gg J_{train}$: cv much better than train
- Simultaneously have high bias and high variance
	- Possible in neural network
	- Part of input had a very complicated model that overfits, then for the rest parts, it doesn't even fit the training data well
	- $J_{train}$ will be high
	- $J_{cv} \gg J_{train}$: cv much better than train

### Regularization and bias/variance
#Regularization  #Underfitting #Overfitting  
How the choice of the regularisation parameter **lambda** affects the bias and variance (overall performance) of the algorithms  
![](Pasted%20image%2020230106193027.png)

> Lambda: control how much you trade off keeping the parameters w small versus fitting the training data well

- The target lambda is, the more the algorithm is trying to keep $w^2$ small, so more weight is giving to this regularization term, and less attention is paying to actually doing well on the training set
- Try to fit a model using lambda equals to 0, 0.01, 0.02, 0.04, 0.08, … 10 and evaluate the cross-validation error

## Learning Curves
#LearningCurve  
![](Pasted%20image%2020230106222103.png)
- When $m_{train}$ gets bigger, cross validation error goes down
- But training error increases: when increasing the training set size, it might be a bit harder for model to fit every single one of the training examples perfectly
- Cross validation error will be typically higher than the training error, as we fit the parameters to the training set, so we can expect it to do better

### How to Plot
- We can train a model with different subsets (100, 200, etc..) of training sets, but this is computationally quite expensive
- Having mental visual picture in the head

![](Pasted%20image%2020230106223709.png)

### High Bias (Underfitting)
- Big gap between the baseline level performance and training error
- Getting more data **will not (by itself) help much** as the curve flattened out
- We shall fit a more complex function

### High Variance (Overfitting)
- Cross validation error is much higher than the training error
- Training error can be even lower than the human level of performance, as we are overfitting the training set
- Getting more data **is likely to help**
