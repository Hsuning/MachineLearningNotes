# Multi-label Classification

## Definition
- A single input can have multiple labels
- ex: detect what is in front of a car. There might be bus, car and pedestrian
- The target Y is a vector of N numbers

## Treat as N completely separate ML problems
- build one binary classification neural network per label
- one to detect whether there is a bus, then one for car, then one for pedestrian

## Train a single NN to simultaneously detect all labels
- The final output layer would have 3 output neurons and will output 3 numbers
- As we are solving 3 binary classification problems, so we can just use a #SigmoidFunction for each of these 3 nodes in the output layer