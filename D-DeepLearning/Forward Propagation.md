# Forward Propagation
Inference: making predictions

#ForwardPropagation 

## Concept
Forward propagation means the computation goes in the forward direction from left to right, start from input to output $\vec{x} --> \vec{a}^{[1]} --> \vec{a}^{[2]} --> a^{[3]}$

![](Pasted%20image%2020221212144551.png)

### Example
- Handwritten digit recognition 0 and 1
- binary classification, to predict the probability of being a handwritten 1

### Architecture
- 3 layers networks
- Architecture: input layer 0 -> layer 1 (25 units) -> layer 2 (15 units) -> layer 3 (1 units)
- $f(x)$: denote the function computed by the neural network
- forward propagation: the computation in the forward direction from left to right, start from input to output $\vec{x} --> \vec{a}^{[1]} --> \vec{a}^{[2]} --> a^{[3]}$

### Layer 0, input layer
- 8 * 8 images = a grid / matrix of 8x8 or 64 pixel intensity values
- 255 denotes a bright white pixel and 0 denote a black pixel
- activation $a^0$ is just the input feature value $\vec{x}$ 

### Layer 1, 1st computation
- 25 hidden units / neurons
- Go from $\vec{x} --> \vec{a}^{[1]}$ ,  get a vector of 25 values

![](Pasted%20image%2020221212142621.png)

### Layer 2,  2nd computation
- 15 hidden units
- Go from $\vec{a}^{[1]} --> \vec{a}^{[2]}$, get a vector of 15 values
![](Pasted%20image%2020221212142913.png)

### Layer 3, output layer, final computation 
- 1 unit
- Go from $\vec{a}^{[2]} --> a^{[3]}$, the final activation value is a scaler as only 1 unit
- Can threshold the output at 0.5 (certain value) to come up with a binary classification

![](Pasted%20image%2020221212142940.png) 

## Inference with Tensorflow
- Inputs: temperature and duration
- Whether or not this setting will result in a good coffee or not
- Dense type layer
![](Pasted%20image%2020221212154022.png)

```
x = np.array([[200.0, 17.0]])

# layer_1 is a function
layer_1 = Dense(units=3, activation='sigmoid')
# Apply function to x
a1 = layer_1(x) # a list of 3 numbers as 3 units

layer_2 = Dense(units=1, activation='sigmoid')
a2 = layer_2(a1)

if a2 >= 0.5:
	yhat = 1
else:
	yhat = 0
```


## Forward Prop in a single layer from scratch
![](Pasted%20image%2020221213112417.png)

To avoid hard coding, more general implementation
![](Pasted%20image%2020221213113133.png)
A function to implement a dense layer (a single layer)
- inputs the activations from the previous layer
- given the parameters for the current layer, it returns the activations for the next layer





