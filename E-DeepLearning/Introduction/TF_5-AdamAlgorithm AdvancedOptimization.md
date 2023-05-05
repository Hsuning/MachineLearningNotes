
## Adam algorithm
#AdamOptimizer
#AdamAlgorithm #GradientDescent 
![](Pasted%20image%2020230104110419.png)
- Adam: Adaptive Moment Estimation
- Automatically increase/decrease alpha, make it take bigger/smaller steps and get to the minimum faster
	-  If $w_j$ or $b$ keeps moving in same direction, increase $a_j$
	-  If $w_j$ or $b$ keeps oscillating, reduce $a_j$
- Not just one alpha: uses a different learning rate for every single parameter of the model


## TF Implementation

The model is exactly the same as before
```
# Model
model = Sequential([
	Dense(units=25, activation='relu'),
	Dense(units=15, activation='relu'),
	Dense(units=10, activation='linear')
])
```

The way we compile the model is with one extra argument
- Optimizer we want to use: adam optimization algorithm with some default initial learning rate alpha
> Worth trying a few values for this initial, with larger / smaller values to see what gives us the fastest learning performance
```
model.compile(
	optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
	loss=SparseCategoricalCrossentropy(from_logits=True)
) 
```