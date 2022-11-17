# Learning Rate
```toc
```
## Concept
Take #LearningRate $\alpha$ (alpha) in #GradientDescent as example:

$$
\begin{align*} \text{repeat}&\text{ until convergence:} \; \lbrace \newline\;
& w_j := w_j - \alpha \frac{\partial J(\mathbf{w},b)}{\partial w_j} \tag{1} \; & \text{for j = 0..n-1}\newline
&b\ \ := b - \alpha \frac{\partial J(\mathbf{w},b)}{\partial b} \newline \rbrace
\end{align*}$$
- The learning rate controls the size of the update to the parameters
- It is usually a small positive number between 0 and 1
- 0-tiny baby step, 1-huge step-
- It is generally shared by all the parameters

## Identify Problem when choosing learning rate
- cost consistently increases (it shall be decreasing)
- cost function goes up and down

## Cause of problem
- bugs in code -> like update function is not minus #PartialDerivative
- learning rate is too large -> overshoot the optimal value at each iteration, and as a result, cost ends up increasing rather than approaching the minimum

## Solution
- Use a smaller learning rate
- Select a very small $\alpha$, see if cost should decrease on every iteration -> if still increase, then there is a bug

## Best Practices
- Try a range of values for the learning rate:
	- 0.001 -> 0.003 -> , 0.01, 0.1, 1 =>
	- try a value 3X bigger than previous
- Plot the cost function with number of iterations
- Pick the largest possible value (or slightly smaller) that decrease the learning rate rapidly
- A learning rate of 0.1 is a good start for regression with normalized features.
