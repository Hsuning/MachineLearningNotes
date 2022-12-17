#Tensorflow 
```
layer_1 = Dense(units=3, activation='sigmoid')
layer_2 = Dense(units=1, activation='sigmoid')

model = Sequential([layer_1, layer_2])

# store data in matrix
x = np.array([...]) # 4x2
y = np.array([1, 0, 0, 1]) # 1d

model.compile(...)
model.fit(x, y)

# do inference
model.predict(x_new)
```
- Sequential:  #TFSequential
	- input
		- X, y in numpy array
	- create a new network by sequentially stringing together these two layers 
- model.compile(...)
- model.fix(X, y):
	- take this neural network and train it on the data x and y

In practice, we would just put the layers directly into the sequential function:
```
model = Sequential([
	Dense(units=25, activation='sigmoid'),
	Dense(units=15, activation='sigmoid'),
	Dense(units=1, activation='sigmoid'),
])
```
