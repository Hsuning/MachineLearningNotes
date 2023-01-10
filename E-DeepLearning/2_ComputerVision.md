# 3_ComputerVision
#ComputerVision

```toc
```
## Face Recognition
![](Pasted%20image%2020221211154438.png)

### Input
- Image = 1000 * 1000 pixels = 1000 * 1000 matrix / grid
- Fro grey, values from 0 to 255, pixel brightness values
- When unrolling them into vector, we will get a list / vector of 1 million values
- $\vec{x}$ = a list / vector of 1 million values

### Architecture
How to train a neural network that takes this vector as input, and output the identity of the person in the picture?
- first hidden layer: looking for very short lines or very short edges
	- one for a little vertical line or a vertical edge
	- one for a oriented line
- second hidden layer: group together lots of little short lines and edges, to look for parts or faces - eyes, noses, botton of noses
- third hidden layer: aggregating different parts of faces to detect presence or absence of larger coarser face shapes, detect how much the face corresponds to different face shapes
- output layer: determine the identify of the person picture

> Activations are higher level features  
> Neural Network can learn theses feature detectors at the different hidden layers all by itself

- We can use same architecture for car classification. Just feed with different data and NN will automatically learn to detect very different features and make prediction
