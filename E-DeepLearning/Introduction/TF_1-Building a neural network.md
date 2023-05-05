Based on the architecture described here: [4_Forward Propagation](4_Forward%20Propagation.md)

```toc
```

![](Pasted%20image%2020221228162130.png)


## Step 0 - Normalisation Layer
- The neural network will fit the weights to the data.
- It will proceed more quickly if the data is normalised
- **Keras normalization layer** will shift and scale inputs into a distribution centered around 0 with standard deviation 1
> This is not a layer in the model

```
# Create a Normalization layer
norm_l = tf.keras.layers.Normalization(axis=-1)

# Adapt the data => learn the mean and variance of the data set
norm_l.adapt(X) 

# Normalize the data
Xn = norm_l(X)
```

## Step 1 - Create the model
- Specify the architecture of neural network
- Sequentially string together the layers of the neural network
- How to compute for the inference
```
tf.random.set_seed(1234)  # applied to achieve consistent results
model = Sequential(
    [
        tf.keras.Input(shape=(2,)),
        Dense(25, activation='sigmoid', name='layer1'),
        Dense(15, activation='sigmoid', name='layer2'),
        Dense(1, activation='sigmoid', name='layer3')
     ]
)

```
- Sequential:  #TFSequential
	- input = X, y in numpy array
	- create a new network by sequentially stringing together these layers 
- tf.keras.Input(shape=(2,)), :  
	- specifies the expected shape of the input, size the weights and bias parameters at this point. 
	- This is useful when exploring Tensorflow models
	- Can be omitted in practice and Tensorflow will size the network parameters when the input data is specified in the `model.fit` statement.
- *Including the sigmoid activation in the final layer*
	- Not considered best practice
	- It would instead be accounted for in the loss which improves numerical stability

- `model.summary()` provides a description of the network:
- The weights 𝑊 : of size (number of features in input, number of units in the layer)
- The bias b: of size the number of units in the layer

## Step 3 - Compile
- Compile the model using a specific loss function
```
from tensorflow.keras.lossess import BinaryCrossentropy

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)
```
- `model.compile`: 
	- defines a loss function
	- specifies a compile optimization => run gradient descent to fit the weights of the model to the training data
- `BinaryCrossentropy()`: same as logistic loss
- Keras is a separate library but was integrated into TF
- `MeanSquaredError()` : if we want to predict numbers 

## Step 4 - Fit
- Fit the model that we specified in step 2 using the loss of cost function in step 3, to the dataset X, y. 
- Compute derivatives for gradient descent using 'backpropagation'

```
model.fit(
    Xt, Yt,            
    epochs=10,
)
```
- `model.fit`:
	- take this neural network and train it on the data x and y
	- runs gradient descent and fits the weights to the data, try to minimize the cost

```
Epoch 1/10
6250/6250 [==============================] - 6s 910us/step - loss: 0.1782
```
- #Epochs: number of steps in gradient descent, specify that the entire data set should be applied during training X times
- #Batches: the training data set is broken into batches. The default size of a batch in Tensorflow is 32 (200000/32 = 6250 batches)


## Step 5 - Prediction

```
X_test = np.array([
    [200,13.9],  # postive example
    [200,17]])   # negative example
X_testn = norm_l(X_test) # normalize again
predictions = model.predict(X_testn)
print("predictions = \n", predictions)
```
- Input data: (m,2), m is the number of examples and two features
- Must normalize test data as training test
  
```
yhat = np.zeros_like(predictions)
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0
print(f"decisions = \n{yhat}")
```
- apply threshold to convert the probability to a decision


