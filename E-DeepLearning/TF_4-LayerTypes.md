
## Dense Layer
#DenseLayer 
![](Pasted%20image%2020230104163551.png)
- Each neuron output is a function of all the activation outputs of the previous layer
- f(all the output values)


## Convolutional Layer
#ConvolutionalLayer
![](Pasted%20image%2020230104164106.png)
- Input x = handwritten digit 9
- Each neuron only looks at part of the previous layer's inputs
- A hidden layer: compute different activations as functions of this input image x. Each neuron only looks at the pixels in a limited region of the image (1x1, 2x2, 2x1, ...) 
- Benefit:
	- faster computation
	- need less training data (less prone to overfitting)

### Convolutional Neural Network
- Have multiple convolutional layers in a neural network
- With many architectural choices
	- how big is the window of inputs that a single neuron should look at
	- how many neurons should each layer have
	- 
![](Pasted%20image%2020230104164809.png)
**Example**
Classification of EKG signals or electrocardiograms, try to diagnose if a patient may have a heart issue
- input: 1-D input, 100 numbers corresponding to the height of a curve at 100 different points of time
- 1st Hidden layer (9 units): looks at a small window of the input 
- 2nd Hidden layer: looks at limited number of activations from previous layer
- Output layer (sigmoid): binary classification