#LinearRegression #MultipleLinearRegression

# Linear Regression with Multiple Variable
```toc
```

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

Perform a vector dot product with ```np.dot()```, #Vectorization

```
b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

def predict(x, w, b): 
    """
    single predict using linear regression
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters   
      b (scalar):             model parameter 
      
    Returns:
      p (scalar):  prediction
    """
    p = np.dot(x, w) + b    
    return p    

# Make a prediction
f_wb = predict(x_vec, w_init, b_init)
```

## Cost Function
The equation for the cost function with multiple variables $J(\mathbf{w},b)$ is:

$$
J(\mathbf{w},b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})^2
$$

where:

$$
 f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = \mathbf{w} \cdot \mathbf{x}^{(i)} + b
$$

In contrast to previous labs, $\mathbf{w}$ and $\mathbf{x}^{(i)}$ are #Vector rather than scalars supporting multiple features.

```
def compute_cost(X, y, w, b): 
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):                                
        f_wb_i = np.dot(X[i], w) + b           #(n,)(n,) = scalar (see np.dot)
        cost = cost + (f_wb_i - y[i])**2       #scalar
    cost = cost / (2 * m)                      #scalar    
    return cost

# Compute and display cost using our pre-chosen optimal parameters. 
cost = compute_cost(X_train, y_train, w_init, b_init)
```

## #GradientDescent
- Vector notation
- Parameter
	- $\vec{w}=[w_1,w_2,w_3,w_4,...,w_n]$
	- b = still a number
- Model = $f_{\vec{w}, b}(\vec{x})=\vec{w}\cdot\vec{x}+b$
- Cost function = $J(\vec{w}, b)$
- Gradient descent:

$$
\begin{align*} \text{repeat}&\text{ until convergence:} \; \lbrace \newline\;
& w_j = w_j - \alpha \frac{\partial J(\mathbf{w},b)}{\partial w_j} \; & \text{for j = 0..n-1}\newline
&b\ \ = b - \alpha \frac{\partial J(\mathbf{w},b)}{\partial b} \newline \rbrace
\end{align*}$$
where, n is the number of features, parameters $w_j$, $b$, are updated simultaneously and where  
$$

\begin{align}  
\frac{\partial J(\mathbf{w},b)}{\partial w_j} &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})x_{j}^{(i)} \\  
\frac{\partial J(\mathbf{w},b)}{\partial b} &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})  
\end{align}

$$
- m is the number of training examples in the data set
- $f_{\mathbf{w},b}(\mathbf{x}^{(i)})$ is the model's prediction, while $y^{(i)}$ is the target value
- Every feature has its $w$
- The #DerivativeTerm is based on $w_j$ in $\frac{d}{dw_j}$

Derivative :
```
def compute_gradient(X, y, w, b):
		"""
		Computes the gradient for linear regression
		Args:
			X (ndarray (m,n)): Data, m examples with n features
			y (ndarray (m,)) : target values
			w (ndarray (n,)) : model parameters  
			b (scalar) : model parameter
			
		Returns:
			dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
			dj_db (scalar): The gradient of the cost w.r.t. the parameter b.
		"""
		m,n = X.shape  #(number of examples, number of features)
		dj_dw = np.zeros((n,))
		dj_db = 0.

		for i in range(m): # outer loop over all m examples
				err = (np.dot(X[i], w) + b) - y[i]

# Err = Model - target_variable
				for j in range(n): # second loop over all n features
						dj_dw[j] = dj_dw[j] + err * X[i, j] # for each w_j  
				dj_db = dj_db + err
		dj_dw = dj_dw / m
		dj_db = dj_db / m
				
		return dj_db, dj_dw
```

Gradient:
```
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
		"""
		Performs batch gradient descent to learn theta. Updates theta by taking
		num_iters gradient steps with learning rate alpha
		
		Args:
			X (ndarray (m,n)) : Data, m examples with n features
			y (ndarray (m,)) : target values
			w_in (ndarray (n,)) : initial model parameters  
			b_in (scalar) : initial model parameter
			cost_function : function to compute cost
			gradient_function : function to compute the gradient
			alpha (float) : Learning rate
			num_iters (int) : number of iterations to run gradient descent
			
		Returns:
			w (ndarray (n,)) : Updated values of parameters
			b (scalar) : Updated value of parameter
			"""
		

# An Array to Store Cost J and W's at Each Iteration Primarily for Graphing Later
		J_history = []
		w = copy.deepcopy(w_in)  #avoid modifying global w within function
		b = b_in
		
		for i in range(num_iters):

# Calculate the Gradient and Update the Parameters
				dj_db, dj_dw = gradient_function(X, y, w, b) ##None

# Update Parameters Using W, B, Alpha and Gradient
				w = w - alpha * dj_dw ##None
				b = b - alpha * dj_db ##None
			

# Save Cost J at Each Iteration
				if i<100000: # prevent resource exhaustion
						J_history.append( cost_function(X, y, w, b))

# Print Cost Every at Intervals 10 Times or as Many Iterations if < 10
				if i% math.ceil(num_iters / 10) == 0:
						print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f} ")
				
		return w, b, J_history #return final w,b and J history for graphing
```

## Normal equation #NormalEquation
- An alternative to gradient descent, for finding w and b for linear regression
- Use advanced linear algebra library
- **Only for linear regression**, so doesn't generalize to other learning algorithm
- Solve for w, b without iterations, but slow when number of features is large (> 10,000)
- Normal equation method may be used in machine learning libraries that implement linear regression
- **Gradient descent** is the recommended method for finding parameters $w,b$ --> **often a better way to get the job done**

## Utilize Scikit-learn to Implement Linear Regression Using Gradient Descent
C1_W2_Lab05_Sklearn_GD_Soln

```
from sklearn.linear_model import SGDRegressor

sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, y_train)
print(sgdr)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")
```
- Gradient descent regression model
- performs best with normalized inputs

```
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)
print(f"Peak to Peak range by column in Raw X:{np.ptp(X_train,axis=0)}")
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")
```
- perform #ZScoreNormalization, also called standard score

```
b_norm = sgdr.intercept_
w_norm = sgdr.coef_

print(f"model parameters: w: {w_norm}, b:{b_norm}")
print( "model parameters from previous lab: w: [110.56 -21.27 -32.71 -37.97], b: 363.16")
```
- the parameters are associated with the normalized input data

## Utilize Scikit-learn to Implement Linear Regression Using a Close Form Solution Based on Normal Equation
- #NormalEquation
- obtain the optimal parameters by just using a formula that includes a few matrix multiplications and inversions
- work well on smaller data sets
- Does not require normalization

C1_W2_Lab06_Sklearn_Normal_Soln

```
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()
#X must be a 2-D Matrix
linear_model.fit(X_train.reshape(-1, 1), y_train)

b = linear_model.intercept_
w = linear_model.coef_
print(f"w = {w:}, b = {b:0.2f}")
print(f"'manual' prediction: f_wb = wx+b : {1200*w + b}")

y_pred = linear_model.predict(X_train.reshape(-1, 1))

```
- a closed-form linear regression
