# TransferLearning
```toc
```

## Concept
![](Pasted%20image%2020230108112306.png)

- a technique that lets us **use data from a different task** to help on our application
- **Supervised pretraining** #SupervisedPretraining
	- train a neural network on a very large data set of not quite related task
	- By learning to recognise cat, dog, etc, the model have learned some plausible sets of **parameters** for the earlier layers for processing image inputs

	> Lot of trained neural networks are available on the internet

- **Fine tuning**
	- Make a copy of supervised pretraining neural network
	- By transferring these **parameters** from the first layers (except final output layer), the new neural network starts off with the parameters in a much better place
	- For the last output layer, modify it to suit the specific application and come up with new parameters that train from scratch
	- Then run an optimisation algorithm (gradient descent, atom optimisation) further to fine-tune the model to suit the specific application, so end up at a pretty good model
- 2 options:
	- ( very small training set): only train output layers parameters
	- (larger training set) train all parameters, but the first N layers would be initialised using the values that we have trained on top

## How it Works?
- Lear to detect generic features
- Download neural network parameters pretrained on a large dataset with **same input type** (e.g., images, audio, text) as your application (or train your own)
- Further train (fine tune) the network on your own data. You can use a much smaller data to fine-tune the model (1000, 50)
- GPT3, Bert: transfer learning
- Example: training a neural network to detect different objects from images
	- First layer: group pixels to detect edges, low level features
	- Second layer: group edges to detect corner
	- Third layer: learn to detect more complex but still generic shapes (curves/basic shapes)

> The first 3 layers learns to detect generic features of images
