#ActivationFunctions

```toc
```

## Why do we need activation functions?
- If we only use linear activation functions for all neurons, this is equivalent to linear regression
- Even if the output activation is #SigmoidFunction => equivalent to logistic regression
- So the big neural network is just same as linear regression or logistic regression
- don't use linear activations in hidden layers (use #ReLuFunction )
- so we need activation function other than linear activations

## Activation Functions
- The output may be binary (0, 1), a number between 0 and 1, or any non-negative numbers
- The activation function is the key to the neural network to achieve nonlinear output. In simple terms, the activation function is a switch of a perceptron, which determines the size and positive or negative of the output value.

![](Pasted%20image%2020221230102615.png)

### Linear Activation Function: g(z) = Z
#LinearActivationFunction 
- No activation function
- A linear function of a linear function is itself a linear function
- If we chain several linear transformations, all we get is a linear transformation. Chaining two linear functions gives you another linear function. If you don't have some non-linearity between layers, then even a deep stack of layers is equivalent to a single layer : you cannot solve very complex problems with that.

### Logistic Function / Sigmoid : σ(z) = 1 / (1 + exp(–z)) 
#LogisticFunction #SigmoidFunction #BinaryCrossentropy
- S-shaped, continuous, and differentiable
- Has a well-defined nonzero derivative everywhere, allowing Gradient Descent to make some progress at every step. Vanishing gradient problem #VanisingGradientProblem
- Non-linear Output from 0 to 1
- can be used as the output layer in the binary classification, which the output represents the probability
- It is easy to cause the problem of #GradientVanishing. The value of the derivative is always less than 1 and always changes around 0, which is very easy to cause the problem of gradient vanishing in the deep neural network.
- not 0 as the axis of symmetry (this is improved in the tahn function)

### Rectified Linear Unit Function : ReLU(z) = max(0, z)
#ReLuFunction
- Rectified Linear Unit
- continuous but not differentiable at z = 0 (the slope changes abruptly => make Gradient Descent bounce around)
- its derivative is 0 for z < 0, and 1 for z > 0
- However, in practice it works very well and has the advantage of being fast to compute.
- Does not have a maximum output value > help reduce some issues during Gradient Descent, ex : vanishing gradient

### Hyperbolic Tangent Function : tanh(z) = 2σ(2z) – 1
- like the logistic function :  S-shaped, continuous, and differentiable
- but output value ranges from -1 to 1
- tends to make each layer's output more or less centered around 0 at the beginning of training, often helps speed up convergence.
- vanishing gradient problem

### Softmax Activation

### Leaky ReLU activation function $f(x)=max(0.01 * x , x)$
- Return x if positive input
- but return a really small value if negative value

Tannage, leaky, switch


## Choosing activation functions
- different activation functions for different neurons

### Output layer 
- will often have 1 choice, depending on the target label y
- `activation='sigmoid'` for binary classification: #SigmoidFunction 
- `activation='linear'` for regression (stock price) if y can be <=0 or >= 0, #LinearActivationFunction
- `activation='relu'` for regression (house price) if y >= 0 , #ReLuFunction 

### Hidden Layer
- `activation='relu'`  as default
- #ReLuFunction : most common choice, faster to compute, not flat
- #SigmoidFunction : rare, goes flat in two places and gradient descent can be really slow. only if binary classification problem

