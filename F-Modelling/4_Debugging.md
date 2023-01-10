# 4_Debugging
```toc
```

## Debugging a Learning Algorithm
- Find a choice where to invest your time
- Debugging takes a lifetime to master

### If High Variance (Overfitting)
- Get more training examples [AddingData](AddingData.md) improves generalization
- Simplify the model
	- Try smaller sets of features
		- reduce the _complexity_, too many features gives the algorithm too much _flexibility_ to fit very complicated models
	- Try increasing #LearningRate  
		- force algorithm to fit a smoother function

### If High Bias (Underfitting)
- Make model more powerful
	- Try getting additional features
		- the algorithm will not do well until you add right features
	- Try adding #PolynomialFeatures
	- Try decreasing #LearningRate
		- Use a lower value, pay less attention on regularisation



## Bias / Variance and Neural Networks
- Neural network = a new way to address both high bias and variance
- Tradeoff between bias and variance => not too simple but not too complex model
- Neural networks are **low bias** machines => if a neural network is large enough, and the training data is not too large, it can almost always fit the training set => without the need of trade off  
![](Pasted%20image%2020230107130840.png)
 

### Regularize a Neural Network
- It hardly ever hurts to have a larger neural network so long as **regularisation** is chosen appropriately
 - A large neural network will usually do as well or better than a smaller one

 > The only pain would be a large neural network is computationally expensive and will slow down the training and inference process

- don't regularise parameter b in neural network
- `kernel_regularizer`
	- L2(0.01): 0.01 is the value of lambda
	- You can use different value of lambda for different layers  
![](Pasted%20image%2020230107131257.png)
