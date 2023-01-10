# Feature Scaling
#FeatureScaling
```toc
```



## Problem Statement
- We can have a very large or very small range of values of a feature
- A good model will learn to choose a relatively small parameter value (like 0.1) when a possible range of values of a feature is large
- When look at the data distribution between two variables with different scales, one axis might have a much larger scale / range of values compared to another
- In cost function, the contours form ellipses that are shorter one side and longer on the other (skinny, tall), and as a result, very small change to w can have a very large impact on the prediction and cost j.
- Cause gradient to run slowly, as skinny gradien descent may end up bouncing back and forth for a long time

## Solution
- perform some transformation to scale the features
- all the features uses comparable ranges of values (like 0 to 1)

![[Pasted image 20221114143615.png]]

## Scaling Method

### Divide by Maximum
- $x_{1, scaled} = \frac{x_1}{max\_of\_x1}$
- Divide each positive feature by its maximum value
- Normalize features to the range of 0 and 1
- Works for positive features
- Simple to use

### Rescale by Both Minimum and Maximum
- Rescale each feature by both its minimum and maximum values (x-min)/(max-min)
- works for any features
- Normalize features to the range of 0 and 1

### Mean Normalization
- Both of them are centred around 0
- Normalize features to the range of -1 and 1
- First find average value $\mu_i$
- $x_i := \dfrac{x_i - \mu_i}{max - min}$

### Z-score normalization
#ZScoreNormalization
- less common
- Normalize feature to have a mean of 0 and a standard deviation of 1
- To implement z-score normalization, adjust your input values as shown in this formula:  

$$
x^{(i)}_j = \dfrac{x^{(i)}_j - \mu_j}{\sigma_j}
$$

- $j$ selects a feature or a column in the $\mathbf{X}$ matrix
- $µ_j$ is the mean of all the values for feature (j)
- $\sigma_j$ is the standard deviation of feature (j)

$$
\begin{align}
\mu_j &= \frac{1}{m} \sum_{i=0}^{m-1} x^{(i)}_j\\
\sigma^2_j &= \frac{1}{m} \sum_{i=0}^{m-1} (x^{(i)}_j - \mu_j)^2 \
\end{align}
$$

>**Implementation Note:** it is important to store the mean value and the standard deviation used for the computations. As we must normalize new data using the same mean and standard deviation that we had previously computed from the training set.

```
def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column
    
    Args:
      X (ndarray (m,n))     : input data, m examples, n features
      
    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    # find the mean of each column/feature
    mu     = np.mean(X, axis=0)                 # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma  = np.std(X, axis=0)                  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma      

    return (X_norm, mu, sigma)
 
#check our work
#from sklearn.preprocessing import scale
#scale(X_orig, axis=0, with_mean=True, with_std=True, copy=True)
```

![[Pasted image 20221115221512.png]]  
The plot above shows the relationship between two features, "age" and "size(sqft)".
- Left: Unnormalized: The range of values or the **variance** of the 'size(sqft)' feature is much larger than that of age
- Middle: The first step removes the mean or average value from each feature. This leaves features that are centered around zero. It's difficult to see the difference for the 'age' feature, but 'size(sqft)' is clearly around zero.
- Right: The second step divides by the variance. This leaves both features centered at zero with a similar scale.

## When to Scale Your Feature

### Acceptable Ranges of Values
- $-1 <= x_j <= 1$
- $-3 <= x_j <= 13$
- $-0.3 <= x_j <= 0.3$

### No Need for Rescaling
- $0 <= x_j <= 3$
- $-2 <= x_j <= 0.5$
- May need to rescale if range is too large range or too small
	- $-100 <= x_j <= 100$
	- $-0.001 <= x_j <= 0.001$
	- $98.6 <= x_j <= 105$

### The Feature Rescaling is Almost Never Any Harm, so We Can Just Carry it Out
