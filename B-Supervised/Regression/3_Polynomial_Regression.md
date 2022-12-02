# Polynomial Regression
#PolynomialRegression  
Combined with #MultipleLinearRegression and #FeatureEngineering

```toc
```

## Problem
- When our features/data are **non-linear** or are **combinations of features**
	- Ex: housing prices do not tend to be linear with living area but penalize very small or very large houses resulting in a curve that fit the data
- #Machinery: ability to modify the parameters $\mathbf{w}, \mathbf{b}$ to fit the equation to the training data. However, no amount of adjusting of parameters will achieve a fit to a non-linear curve

## Solution
- use #FeatureEngineering and #PolynomialRegression which allows you to use the machinery of #LinearRegression to fit very complicated, even very non-linear function  
![[Pasted image 20221117123604.png]]

## Polynomial Features
- took an optional feature x and raised it to the power of 2 or 3 or any other power
	- quadratic function: $x^2$ -> $y= w_0x_0^2 + b$
		
	- cubic function: $x^3$ -> $y= w_0x_0^3 + b$
- squared root $\sqrt{x}$

## Selecting Features
- Add a variety of potential features to try and find the most useful features
	- Ex: try $y=w_0x_0 + w_1x_1^2 + w_2x_2^3+b$

### 1-Gradient Descent Method
- Gradient descent is picking the 'correct' features for us by emphasizing its associated parameter
	- model after training $0.08x + 0.54x^2 + 0.03x^3 + 0.0106$
- the **weight** associated with the $x^2$ feature is **much larger** than the weights for $x$ or $x^3$ as **it is the most useful in fitting the data**
- The features were re-scaled, so they are comparable to each other
- **Less weight value** implies **less important/correct feature**. When the weight becomes zero or very close to zero, the associated feature would not be useful in fitting the model to the data

### 2-Linear Relative
- The best features will be linear relative to the target
- ![[Pasted image 20221117123349.png]]
- Plot the relation
- It is clear that the 𝑥2 feature mapped against the target value 𝑦 is linear. Linear regression can then easily generate a model using that feature

## Scaling features
#FeatureScaling
- Feature scaling become increasingly important as the polynomial features could range from one to 1000 (1000^2, 1000^3)
- Feature scaling allows this to converge much faster
- must apply feature scaling to convert features into comparable ranges of values
- with feature engineering, even quite complex functions can be modelled
- ![[Pasted image 20221117123941.png]]
