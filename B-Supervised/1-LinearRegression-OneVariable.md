#LinearRegression #LinearRegressionOneVariable #Supervised 
# Linear Regression with one variable 

## Concept
```
             ┌────────────┐
             │training set│   features
             └─────┬──────┘   targets
                   │
                   │
         ┌─────────▼─────────┐
         │learning algorithms│
         └─────────┬─────────┘
                   │
                   │
 size        ┌─────▼──────┐       estimated price
   x  ────►  │f (function)│ ────►  y-hat
             └────────────┘
feature        model             prediction
                                (estimated y)
```
- Also called univariate linear regression
- Only take a single feature x, and output the estimated $\hat y$
- $f(x) = wx+b$
	- linear function
	- $f_{w,b}(x) = wx+b$: we can drop w and b and just write f
	- $w,b$: #Parameter, w for slope, b for shift the time

## Notation
![[supervised-1-3.png]]

```
def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      y (ndarray (m,)): target values
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb

# Make a prediction
w = 200                         
b = 100    
x_i = 1.2  # house with 1200 sqft, units of x are in 1000's
cost_1200sqft = w * x_i + b    
print(f"${cost_1200sqft:.0f} thousand dollars")
```

[Model_Representation](.PythonCodes/Supervised_1_2/C1_W1_Lab03_Model_Representation_Soln.ipynb)

## Cost Function
#CostFunction #MeanSquaredErrorCostFunction #SquaredErrorCostFunction
- Provide a measure of how well our predictions match our training data. Minimizing the cost can provide optimal values of $w, b$ of a model.
- Utilize input training data to fit the parameters $w, b$ by minimizing a measure of the error between our predictions and the actual data $y^{(i)}$. The measure is called the _cost_, $J(w,b)$. In training we measure the cost over all of our training samples X, y.
- Goal --> Find $w,b$: $\hat y^(i)$ is close to $y^{(i)}$ for all $(x^{(i)}, y^{(i)})$
- $J(w,b) = \frac{1}{2m} \sum_{i=1}^m (\hat y - y)^2$
	- $\hat y - y$: error, dirrerence between the target value and the prediction
	- $m$ : number of training examples
	- $\sum$: sum up to measure the error across the entire training set
	- $\frac{1}{m}$ : compute the average squared error so cost function will not automatically get bigger as the training set size gets larger
	- $\frac{1}{2m}$: make calculation a bit neater
	- $J(w,b)$: cost function
- always end up with a bow shape or hammock shape
- 
```
def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0] 
    
    cost_sum = 0 
    for i in range(m): 
        f_wb = w * x[i] + b   
        cost = (f_wb - y[i]) ** 2  
        cost_sum = cost_sum + cost  
    total_cost = (1 / (2 * m)) * cost_sum  

    return total_cost
```
[Cost_function_Representation](.PythonCodes/Supervised_1_2/C1_W1_Lab04_Cost_function_Soln.ipynb)


## General Case
- model: $f_{w,b}(x) = wx+b$
- parameters: $w, b$
- cost function: $J(w,b) = \frac{1}{2m} \sum_{i=1}^m (f_{w,b}(x^{(i)}) - y)^2$ =  $J(w,b) = \frac{1}{2m} \sum_{i=1}^m (wx^{(i)}+b - y^{(i)})^2$ 
- **goal of linear regression**: $minimize_{w,b}J(w,b)$
- visualisation: 3D plot or contour plot (a graphical technique for representing a 3-dimensional surface by plotting constant z slices, called contours, on a 2-dimensional format)
![[supervised-1.png]]
- Choose w, b that the cost is close to the center of the small ellipse (minimum)

## Simplified Case --> b = 0 
- model: $f_{w}(x) = wx$
- parameters: $w$
- cost function: $J(w) = \frac{1}{2m} \sum_{i=1}^m (f_{w}(x^{(i)}) - y)^2$
- **goal of linear regression**: $minimize_{w}J(w)$
- visualisation:
```
                J(w)
         │ Function of w
         │ based on fw(x)
       O │               O
       O │               O
      4 O│              O
         O             O
      3  │O           O
 J(w)    │ O         O
      2  │  O       O
         │   O     O
      1  │    O   O
         │     O O
      0 ─┴──────*─────────── w
         0 0.5 1.0 1.5 2.0 2
```
- Choose w to minimize $J(w)$
	- choose the value of w that causes $J(w)$ to be as small as possible
	- ex: w=1 here (\*)


## Conclusion
```
The goal of linear regression is to find the parameters w or (w, b) that results in the smallest possible value J(w)
```


# Gradient Descent for linear regression
- #GradientDescent with #SquaredErrorCostFunction 
repeat until convergence {
$w=w-\alpha*\frac{d}{dw}J(w,b) = w - \alpha\frac{1}{m}\sum_{i=0}^{m-1}(f_{w,b}(x^{(i)})-y^{(i)})x^{(i)}$
$b = b-\alpha*\frac{d}{db}J(w,b) = b - \alpha\frac{1}{m}\sum_{i=0}^{m-1}(f_{w,b}(x^{(i)})-y^{(i)})$
}
_use i=0 and m-1 in codes_
_update w and b simultaneously_
By the rule of calculus:
![[supervised-1-2.png]]
- As #SquaredErrorCostFunction ( #CovexFunction with **bowl shape**) is used here. 
	- Due to the **bowl shape**, the derivatives will always lead gradient descent toward the bottom where the gradient is zero. 
	- The cost function does not and will never have multiple #LocalMinima. **It has only a single #GlobalMinima and will always converge to this point**. 
- #BatchGradientDescent: each step of gradient descent uses all the training data
- #SubsetsGradientDescent: look at smaller subests of the training data at each update step
We use batch gradient descent for linear regreassion.

- implementing derivative part ($\frac{d*J(w,b)}{dw}$, $\frac{d*J(w,b)}{db}$) above 
```
def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    
    # Number of training examples
    m = x.shape[0]    
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):  
        f_wb = w * x[i] + b 
        dj_dw_i = (f_wb - y[i]) * x[i] 
        dj_db_i = f_wb - y[i] 
        dj_db += dj_db_i
        dj_dw += dj_dw_i 
    dj_dw = dj_dw / m 
    dj_db = dj_db / m 
        
    return dj_dw, dj_db
```

- Gradient descent, utilizing compute_gradient and compute_cost
```
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function): 
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x (ndarray (m,))  : Data, m examples 
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters  
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient
      
    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b] 
      """
    
    w = copy.deepcopy(w_in) # avoid modifying global w_in
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []
    b = b_in
    w = w_in
    
    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w, b)     

        # Update Parameters using equation (3) above
        b = b - alpha * dj_db                            
        w = w - alpha * dj_dw                            

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append(cost_function(x, y, w , b))
            p_history.append([w,b])
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
 
    return w, b, J_history, p_history #return w and J,w history for graphing
```
[Gradien_Descent_Representation](.PythonCodes/Supervised_1_2/C1_W1_Lab05_Gradient_Descent_Soln.ipynb)