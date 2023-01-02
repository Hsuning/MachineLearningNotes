# 4_NueralNetworkArchitecture
```toc
```

How to construct a layer of neurons, and take those building blocks to form a large neural network.

Taken a binary classification as example
- target: If a t-shirt will be a top selling
- data: price and top seller or not
- goals: inventory level, marketing campaign  

$$
x \text{ price} --> o(neuron) --> a = \text{ probability of being topseller}
$$  

![](Pasted%20image%2020221211110352.png)

## Neural Network
#NeuralNetwork
$$
\begin{flalign}
&\vec{x} = \text{ input layer} --> \\
&\text{hidden layer that output } \vec{a} \text{ = 3 numbers / activation values} --> \\
&\text{output layer that output a = probability of being topseller}
\end{flalign}
$$

- an oversimplified model of the human brain
- learn its own features from inputs, that makes it easier to make accurate predictions
- automatically learn the features without the need of manual feature engineering

### Input
- (X, y)

### Output / Activation / Function
#ActivationFunctions
- $a=f(x)=\frac{1}{1+e^{-(wx+b)}}$  
- $a$ = activation, how much a **neuron** is sending a high output to other neurons  
- degree that the biological neuron is sending a high output value or sending many electrical impulses to other neurons to the downstream from it

### Neuron
#Neuron
- logistic model can be a simple neuron  
- tiny little computer, the only job is to input one number or a few numbers, and output one/a few other numbers (probability of being top seller)
- each neuron take input - same or similar features, and output a number

### Layer
#Layer
- a grouping of neurons
- can have 1 to N neurons
- a neuron network has a few layers, each layer inputs a vector and outputs another vector of numbers

#### Input Layer
- with X, y
- Input features: price, shipping cost, marketing, material
- Input for NN is a vector $\vec{x}$ of 4 features

#### Hidden Layer
- Learn its own features from inputs, that makes it easier to make accurate predictions
- Something that are hidden in the dataset, ex: affordability, awareness, perceived quality

#### Output Layer
- the output of this layer is the output probability predicted by neural network

## Practices
- Give the layers different number: layer 1, layer 2. Use superscript square bracket $[1]$, to index into different layers
- Input layer = layer 0
- hidden layer = 1… (n-1)
- output layer = n
- Steps:
	- Each layer inputs a vector of numbers and applies a bunch of logistic regression units to it
	- Then computes another vector of numbers that then gets passed from layer to layer until you get the final output layer
	- The final output layer computes the prediction of the neural network , that you can either threshold at 0.5 or not to come up with the final prediction
- Typical choice: start with more hidden units initially, and then the number of hidden units decreases as we get closer to the output layer
- Choosing right number of hidden layers and hidden units per layers can have an impact on the performance of learning algorithms
	- Ex: #MultiLayerPerceptron  
![](Pasted%20image%2020221211110654.png)


## Architecture

### Input
$\vec{x}$: a vector of 4 numbers

### Layer 1: First Hidden Layer
![](Pasted%20image%2020221211163141.png)
- each neuron
	- implement a  #LogisticRegression function
	- has 2 parameters $\vec{w}_1, b_1$
	- output some activation value $a_1$, like 0.3
- $\vec{a}$: output of first hidden layer, which is a vector of activation values $a_1, a_2, a_3$
	- $\vec{a}^{[1]}$: output of layer 1

### Layer 2: Output Layer
![](Pasted%20image%2020221211163438.png)
- the input of layer 2 is the output of layer 1
- as it only have 1 neuron, it computes $\vec{a}^{[1]}$ use #LogisticRegression function / #SigmoidFunction
- $a^{[2]}$ the output of this layer, only a number
- If we want a binary classification [1 or 0], we can take the number, threshold this at 0.5, so $\hat{y}=1, \text{if a[2] > 0.5, else } \hat{y}=0$


## More Complex Neural Network
![](Pasted%20image%2020221212114004.png)
- This neural network has 4 layers, that includes all the hidden layers and the output layer. We don't count the input layer
- Layer 3 has 3 neurons (3 hidden units)
- $[l]$: superscript square brackets, denote which layer the parameters are associated with
- $\vec{w}_2^{[3]}$ = parameter $\vec{w}$ of 2nd neuron (unit) of layer 3

## General Form of Equation
$$
a_j^{[l]} = g(\vec{w}_j^{[l]} \cdot \vec{a}^{[l-1]} + b_j^{[l]})
$$

- $j$ = unit (neuron)
- $l$ = layer
- $a_j^{[l]}$ = activation value of layer $l$, unit(neuron) $j$
- $\vec{a}^{[l-1]}$ = output of layer $l-1$ (previous layer)
- $\vec{w}_j^{[l]}$ and $b_j^{[l]}$ = parameters w & b of layer $l$ unit $j$
- $g$ = sigmoid function = activation function
- $a^{[0]}$: in order to make all this notation consistent, we also give the input vector x this name, so same equation also works for the first layer
