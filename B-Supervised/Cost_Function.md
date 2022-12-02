# Cost Function
#CostFunction
 - The cost functions differ significantly between linear and logistic regression, but adding regularization to the equations is the same.

## Regularization
#Regularization #RegularizedCostFunction 
- To reduce #Overfitting 
- A modified cost function to apply regularization on learning algorithm
- Concept
	- Make parameters ($W_j$) using smaller values (Ex: 0.01 or 0.001)
	- So the model is simpler and less likely to overfit
	- We can only penalize some parameters. However, imaging that we have 100 features, it would be difficult to define which one we shall penalize. Hence, we can just **Penalize all the features $W_j$** 

- $\frac{\lambda}{2m}\sum_{j=1}^{n}W_j^2$: regularization term
	- n: number of features
	- $\lambda = lambda$ : regularization parameter, need to choose a number
	- Divide by $2m$ to scale all the terms in the same way. It is also easier to choose $\lambda$
	- Including this term encourages gradient descent to minimize the size of the parameters. Note, in this example, the parameter b is not regularized. This is standard practice.
- Algorithm
	- Add a new regularization term
	- No need to penalize b as no much difference.
$$
J ( \vec{w}, b ) = \frac{1}{2m} \sum_{i=1}^m (f_{\vec{w},b}(\vec{x}^{(i)} - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^n W_j^2
$$

![[Pasted image 20221201142120.png]]

### How to choose $\lambda$
#Lambda
- It is import to balance 1st and 2nd terms
	- Minimize first term - mean squared error, to ensure that we fit the training data well
	- Minimize the second term - keep $Wj$ small, to reduce the overfitting
- Value of Lambda:
	- $\lambda = 0$: meaning the algorithm is not using the regularization, so overfitting
	- $\lambda = 10^{10}$: meaning that all the value of $w$ are very closed to 0, hence the algorithm would be $f(x)=b$, and we will get a straight line and underfit the data 


