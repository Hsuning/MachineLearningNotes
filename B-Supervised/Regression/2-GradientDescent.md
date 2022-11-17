# Gradient Descent
#GradientDescent
```toc
```

## Concept
- Automatically find the values of parameters that minimizes the cost function and gives the best model
- Cost function: squared error cost, not squared error costs
- Model: #LinearRegression, or not linear regression (ex: some of the biggest and most complex models in all of AI)

------

## Step
- Have some function $J(w,b)$
- Want $min_{w,b}J(w,b)$
- Start up at a point and get to the bottom of one of valleys (local minima) as efficiently as possible
	- Choose some initial values for $w, b$ --> set initial values (w=0, b=0) --> Choose a starting point at the surface
	- Keep changing $w,b$ to reduce $J(w,b)$ --> spin around and pick the direction of steepest descent (in that direction, a tiny baby little step can take you down faster than any other tiny baby step in any other directions)
	- Repeat the process until we settle at or near a minimum. Possibly we may have >1 minimum.
	- A change in starting point might led to change in local minima
- It can led to #LocalMinima instead of #GlobalMinima
- #GlobalMinima: the point that has the lowest possible value for the cost function  
![[supervised-2.png]]
------

## Algorithm
repeat until convergence {  
$w=w-\alpha*\frac{d}{dw}J(w,b)$  
$b=b-\alpha*\frac{d}{db}J(w,b)$  
}
- $\alpha$
	- #LearningRate #alpha, usually a small positive number between 0 and 1
	- control how big of a step we take downhill
	- 0-tiny baby step, 1-huge step
- $\frac{d}{dw}J(w,b)$
	- #DerivativeTerm #PartialDerivative, come from #Calculus
	- which direction we want to take our baby step
- $\alpha*\frac{d}{dw}J(w,b)$
	- combine #alpha and #DerivativeTerm to define the size of the steps we want to take
- Repeat these 2 updates until convergence (reach the point at a local minimum where the parameters w and b no longer change much with each additional step that you take)

### Correct Implementation --> Simultaneous Update
- Calculate the partial derivatives for all the parameters before updating any of the parameters.
- $tempw=w-\alpha\frac{d}{dw}J(w,b)$
- $tempb=b-\alpha*\frac{d}{db}J(w,b)$
- $w=tempw$
- $b=tempb$

## #DerivativeTerm #PartialDerivative
- $w=w-\alpha*\frac{d}{dw}J(w)$  
![[supervised-2-2.png]]

# #LearningRate
- If $\alpha$ is too small, Gradient descent may be slow.
- If $\alpha$ is too large, Gradient descent may overshoot never reach minimum. Fail to converge, and even diverge  
![[supervised-2-3.png]]
- If our parameters have already brought us to a local minimum ($slope = 0, w = w - a * 0$), then further gradient descent steps do absolutely nothing
- Can reach local minimum with fixed learning rate (without decreasing learning rate) -> Near a local minimum
	- derivative becomes smaller
	- update steps become smaller  
![[supervised-2-4.png]]

## Checking Gradient Descent for Convergence
- learning rate $\alpha$ is one of the key choices
- use **learning curve** to determine  
	![[Pasted image 20221114222442.png]]
	- x = number of iterations (simultanous update), vary a lot between different applications
	- y = cost function
	- If j ever increases after one iteration:
		- alpha is chosen poorly or too large
		- bug in the code
	- curve flattened out = coverage
- possibly use automatic convergence tests
	- let $\epsilon=10^{-3}$ (epsilon=0.001)
	- If $J(\vec{w}, b) < \epsilon$ in one iteration, declare convergence
	- however, choose the right epsilon is difficult, so look at learning curve might be easier
- looking at the solid figure can give you advanced warning if gradient descent is not working well

### Two types of gradient descent
- #BatchGradientDescent: each step of gradient descent uses all the training data
- #SubsetsGradientDescent: look at smaller subsets of the training data at each update step  
We use batch gradient descent for linear regression.
