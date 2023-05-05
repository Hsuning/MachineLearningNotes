```toc
```

# Logistic Regression
#LogisticRegression #BinaryClassification 

## Concept
![[Pasted image 20221118210509.png]]
- Fit a sort of s-shaped curve to data
- Inputs a feature or set of features x and outputs a number between 0 and 1
- Apply the sigmoid function (logistic function) to the familiar linear regression model
- Horizontal axis takes on both negative and positive value, labelled as axis $z$

## Algorithm
$f_{\vec{w},b}(\vec{x}) = g(\vec{w}\cdot\vec{x}+b) = g(z) = \frac{1}{1+e^{-z}}, where:0<g(z)<1$
- $\mathbf{w} \cdot \mathbf{x}$ is the vector dot product: $\mathbf{w} \cdot \mathbf{x} = w_0 x_0 + w_1 x_1$
- g(z): sigmoid function that maps all input values to values between 0 and 1
- $z$
	- the input to the sigmoid function, is the output of a linear regression
	- $z$ can be a scaler or a vector consisting of $m$ values
- $e=2.7$, mathematical constant, us ```np.exp()``` to calculate the exponential
- $e^{-z}$ :
	- when z = 0, $g(z)=\frac{1}{1+1}=0.5$, so it passes the vertical axis at 0.5
	- when z is very large positive number, $e^{-z}$ is a tiny number, and ends up being $\frac{1}{1+very.tiny.number}$, so very close to 1
	- when z is a very large negative number, $\frac{1}{1+giant.number}$, so very close to 0
- start very close to zero, and slowly build up to the value of 1
- pick a threshold, $f(x)>=threshold, \hat{y}=1; f(x)<threshold, \hat{y}=0$

Sigmoid function:
- For large positive values of x, the sigmoid should be close to 1, while for large negative values, the sigmoid should be close to 0. 
- Evaluating `sigmoid(0)` should give you exactly 0.5. 
```
def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
         
    """
	z = np.clip( z, -500, 500 )           # protect against overflow
    g = 1/(1+np.exp(-z))
   
    return g
```


## Output
$f_{\vec{w},b}(\vec{x}) = P(y=1|\vec{x};\vec{w}, b)$

if $f_{\mathbf{w},b}(x) >= 0.5$, predict $y=1$  
if $f_{\mathbf{w},b}(x) < 0.5$, predict $y=0$
- Probability that y is 1, given the input features $\vec{x}$, parameters $w,b$
- $P(y=0)+P(y=1)=1$
- $f_{\vec{w}, b}(\vec{x})=0.7$ means 70% chance that y is 1, so 30% that y is 0
- For a logistic regression model, $z = \mathbf{w} \cdot \mathbf{x} + b$
	- if $\mathbf{w} \cdot \mathbf{x} + b >= 0$, the model predicts $y=1$
	- if $\mathbf{w} \cdot \mathbf{x} + b < 0$, the model predicts $y=0$

## Decision boundary
#DecisionBoundary
- a hpyersurface / line that partitions the underlying vector space into 2 sets, one for each class. The classifier will classify all the points on one side of the decision boundary as belonging to one class and all those on the other side as belonging to the other class.
![[Pasted image 20221121172837.png]]


### Linear
- Model: $f(x) = g(z) = g(w_0x_0+w_1x_1 + b)=\frac{1}{1+e^{-z}}$
- Decision boundary: $z=\vec{w}\cdot\vec{x}+b=0$
- model will predict 1 for features to the right of the line, and 0 for features to the left of the line

### Non-linear
![[Pasted image 20221118220110.png]]
- Use #PolynomialFeatures
- $f(x) = g(z) = g(w_0x_0^2+w_1x_1^2 + b)$
- Decision boundary =
	- $z = x_0^2+x_1^2-1=0$
	- = $x_0^2+x_1^2=1$
- So $\hat{y}=0,\quad \textrm{if} \quad x_0^2+x_1^2 < 1$
- We can come up with even more complex decision boundaries by having even higher order polynomial terms, which predict y equals to 1 inside the shape => $g(z)=w_1x_1+w_2x_2+w_3x_1^2+w_4x_1x_2+w_5x_2^2$


## Cost Function for Logistic Regression
#CostFunction #LogisticRegression 
- How to choose parameters $\vec{w}, b$
- We cannot use #SquaredErrorCostFunction used by linear regression:  
 $f_{wb}(x)$ now has a non-linear component, the sigmoid function:   $f_{w,b}(x^{(i)}) = sigmoid(wx^{(i)} + b )$. We will get a non-convex cost function with lots of local minima


### Logistic Loss Function
#BinaryCrossentropy #LogLoss
- more suited to its non-linear nature where the target is 0 or 1
- Loss #Loss : measure of the difference of a single example to its target value
- Cost #Cost: measure of the losses over the training set

**The cost for a single data point:**
$$\text{loss(model prediction, target value)} = loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)})=$$
$$- \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) \text{, if y(i)=1}$$
$$- \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) \text{, if y(i)=0}$$
**That can be rewritten to be easier to implement:**
    $$loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = (-y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)$$
**As y(i) can have only two values, 0 and 1:**
when $y^{(i)} = 0$, the left-hand term is eliminated:
$$
\begin{align}
loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), 0) &= (-(0) \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - 0\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) \\
&= -\log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)
\end{align}
$$
and when $y^{(i)} = 1$, the right-hand term is eliminated:
$$
\begin{align}
  loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), 1) &=  (-(1) \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - 1\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)\\
  &=  -\log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)
\end{align}
$$
**It uses two separate curves that are similar to the quadratic curve of the squared error loss**
	- when the target is zero (y=0)
	- the target is one (y=1)
	- The sigmoid output is strictly between 0 and 1
![[Pasted image 20221121141914.png]]

The final curve is well suited to gradient descent. 


### Cost Function
- Derived from statistics using a statistical principle called maximum likelihood estimation
$$\begin{align*}
	&& J(\mathbf{w},b) = \\
	&& \frac{1}{m} \sum_{i=0}^{m-1} \left[ loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) \right] = \\
	&& -\frac{1}{m} \sum_{i=1}^{m} [ y^{(i)}log(f_{\vec{w},b} (\vec{x}^{(i)})) + (1 - y^{(i)})log(1 - f_{\vec{w},b} (\vec{x}^{(i)})]
\tag{1}\end{align*}$$

where
* $loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)})$ is the cost for a single data point, which is:
    $$loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = -y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) \tag{2}$$
*  where m is the number of training examples in the data set and:
$$
\begin{align}
  f_{\mathbf{w},b}(\mathbf{x^{(i)}}) &= g(z^{(i)})\tag{3} \\
  z^{(i)} &= \mathbf{w} \cdot \mathbf{x}^{(i)}+ b\tag{4} \\
  g(z^{(i)}) &= \frac{1}{1+e^{-z^{(i)}}}\tag{5} 
\end{align}
$$
```
def compute_cost_logistic(X, y, w, b):
    """
    Computes cost

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
        z_i = np.dot(X[i], w) + b  # w . x + b
        f_wb_i = sigmoid(z_i)
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
             
    cost = cost / m
    return cost

```

## Logistic Gradient Descent #GradientDescent 
- Similar to #GradientDescentForLinearRegression 
- Same concepts compared to #GradientDescentForLinearRegression 
	- Use learning curve to monitor gradient descent
	- Run faster with vectorized implementation
	- Apply same feature scaling #FeatureScaling to speed up the convergence
- Recall the gradient descent algorithm utilizes the gradient calculation:
$$\begin{align*}
&\text{repeat until convergence:} \; \lbrace \\
&  \; \; \;w_j = w_j -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial w_j} \tag{1}  \; & \text{for j := 0..n-1} \\ 
&  \; \; \;  \; \;b = b -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial b} \\
&\rbrace
\end{align*}$$

Where each iteration performs simultaneous updates on $w_j$ for all $j$, where
$$\begin{align*}
\frac{\partial J(\mathbf{w},b)}{\partial w_j}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})x_{j}^{(i)} \tag{2} \\
\frac{\partial J(\mathbf{w},b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}) \tag{3} 
\end{align*}$$
- For a logistic regression model  
    $z = \mathbf{w} \cdot \mathbf{x} + b$  
    $f_{\mathbf{w},b}(x) = g(z)$  
    where $g(z)$ is the sigmoid function:  
    $g(z) = \frac{1}{1+e^{-z}}$   

### Implementation from Scratch using Python
Implementation for equation 2 and 3:
```
def compute_gradient_logistic(X, y, w, b): 
    """
    Computes the gradient for linear regression 
 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
    Returns
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape
    
    # Initialize variable
    dj_dw = np.zeros((n,))                           #(n,)
    dj_db = 0.

    # for each example
    for i in range(m):
        # calculate the error for that example
        f_wb_i = sigmoid(np.dot(X[i],w) + b)          #(n,)(n,)=scalar
        err_i  = f_wb_i  - y[i]                       #scalar
        
        # for each input feature
        for j in range(n):
            # multiply the error by the input feature, and add to the corresponding j element of dj_dw
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]      #scalar
        dj_db = dj_db + err_i
        
    # divide by total number of examples (m)
    dj_dw = dj_dw/m                                   #(n,)
    dj_db = dj_db/m                                   #scalar
        
    return dj_db, dj_dw
```

Implementation for equation 1:
```
def gradient_descent(X, y, w_in, b_in, alpha, num_iters): 
    """
    Performs batch gradient descent
    
    Args:
      X (ndarray (m,n)   : Data, m examples with n features
      y (ndarray (m,))   : target values
      w_in (ndarray (n,)): Initial values of model parameters  
      b_in (scalar)      : Initial values of model parameter
      alpha (float)      : Learning rate
      num_iters (scalar) : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,))   : Updated values of parameters
      b (scalar)         : Updated value of parameter 
    """
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)   

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw  # (n, ) * (n, )
        b = b - alpha * dj_db               
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( compute_cost_logistic(X, y, w, b) )

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")
        
    return w, b, J_history         #return final w,b and J history for graphing
```


### Scikit Learn Model
```
from sklearn.linear_model import LogisticRegression

# Initialize model and fit it on the training data
lr_model = LogisticRegression()
lr_model.fit(X, y)

# Get the prediction
y_pred = lr_model.predict(X)

# Calculate the accuracy
print("Accuracy on training set:", lr_model.score(X, y))
```



#RegularizedLogisticRegression
#Regularization #RegularizedCostFunction 

Recall that for regularized logistic regression, the cost function is of the form
$$J(\mathbf{w},b) = \frac{1}{m}  \sum_{i=0}^{m-1} \left[ -y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) \right] + \frac{\lambda}{2m}  \sum_{j=0}^{n-1} w_j^2$$

Compare this to the cost function without regularization (which you implemented above), which is of the form 

$$ J(\mathbf{w}.b) = \frac{1}{m}\sum_{i=0}^{m-1} \left[ (-y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)\right]$$

The difference is the regularization term, which is $$\frac{\lambda}{2m}  \sum_{j=0}^{n-1} w_j^2$$ 
Note that the $b$ parameter is not regularized.
- Penalizing $W_j$ , preventing them for being too large
- Even fitting a high order polynomial with lots of parameters, you can still get a reasonable decision boundaries

Use #GradientDescentRegularized, same equation