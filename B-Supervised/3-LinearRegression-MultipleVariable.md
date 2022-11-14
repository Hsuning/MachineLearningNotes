#LinearRegression #MultipleLinearRegression
# Linear Regression with Multiple Variable

## Concept
- $x_j$ = $j^{th}$ feature, $j$ = index for feature
- $n$ = number of features
- $\vec{x}^{(i)}$ = feature of $i^{th}$ training example
	- $\vec{x}^{(2)} = [1416, 3, 2, 40]$
- $x_j^{(i)}$ = value of feature $j$ in $i^{th}$ training example


## Algorithm
![[supervised-3-1.png]]
- $f_{w,b}(x) = w_1x_1 + w_2x_2 +... + w_nx_n + b$
	- linear algebra: count from 1
	- $\vec{w}=[w_1,w_2,w_3,w_4,...,w_n]$: parameters of the model
	- b is a number
	- $\vec{x} = [x_1, x_2, x_3, ..., x_n]$: features
=> $f_{\vec{w}, b}(\vec{x})=\vec{w}\cdot\vec{x}+b$


## Vectorization
```
# Parameters and features
w = np.array([1.0, 2.5, -3.3])
b = 4
x = np.array([10, 20, 30])

# Vectorization
f = np.dot(w, x) + b
```
- make code shorter, easier to write and read
- run much faster, as numpy dot function is able to use parallel hardware in our computer

![[Pasted image 20221112094356.png]]
- Step
	- take all the values in w
	- multiple **in parallel** all values in vector x
	- and assign all calculation back to w
	```
	w = w - 0.1 * d   # 0.1 = learning rate, ignore b here
	```
- efficient -> scale to large dataset

[Python_Numpy_Vectorization](.PythonCodes/Supervised_3/)


## #GradientDescent 
- Vector notation
- Parameter 
	- $\vec{w}=[w_1,w_2,w_3,w_4,...,w_n]$
	- b = still a number
- Model = $f_{\vec{w}, b}(\vec{x})=\vec{w}\cdot\vec{x}+b$
- Cost function = $J(\vec{w}, b)$
- Gradient descent
	repeat {
	$w_j = w_j - \alpha\frac{d}{dw_j}J(\vec{w}, b)$
	$b = b - \alpha\frac{d}{db}J(\vec{w}, b)$
	}
![[Pasted image 20221112183051.png]]
- Every feature has its $w$
- The #DerivativeTerm is based on $w_j$ in $\frac{d}{dw_j}$

## Normal equation #NormalEquation
- An alternative to gradient descent, for finding w and b for linear regression
- Use advanced linear algebra library 
- **Only for linear regression**, so doesn't generalize to other learning algorithm
- Slove for w, b without iterations, but slow when number of features is large (> 10,000)
- Normal equation method may be used in machine learning libraries that implement linear regression
- **Gradient descent** is the recommened method for finding parameters $w,b$ --> **often a better way to get the job done**


