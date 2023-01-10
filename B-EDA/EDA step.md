# Exploratory Data Analysis
- Turkey 1977
- look at the data
- Summarize and visualize the data
- Gain valuable intuition and understanding of the project

```toc
```

#Inference
- A (complex) set of procedures for drawing conclusions about large populations based on small samples

## Step 1 - [Data & Variable](Data%20&%20Variable.md)


## Step 2 - Estimates of Location
- An estimate of where most of the data is located

### Mean
- Mean / Average
- Weighted mean / average
	- multiplying each data value by a user-specified weight and dividing their sum by the sum of the weights
	- give a lower weight to highly variable observations (ex: taking the average from multiple sensors and one of the sensors is less accurate, so downweight it)
	- the data collected does not equally represent the different groups that we are interested in measuring (give a higher weight to the values from the groups that were underrepresented)
	- np.average(x, weights=w)
- Trimmed mean / truncated mean
	- the average of all values after dropping a fixed number of extreme values (top and bottom 10%)
	- eliminates the influence of extreme values

### Median
- Median / 50th percentile
- Weighted Median
	- sort the data, with an associated weight for each data value
	- a value such that the sum of the weights is equal for the lower and upper halves of the sorted list
	- wquantiles.median()
- a robust estimate of location, not influenced by outliers

### Outliers
- Any value that is very distant from the other values in a data set
- Robust / resistant : not sensitive to extreme values
- Possibly the data value is still valid
- But often the result of data errors = mixing data of different units, bad reading from a sensor
- Mean = poor estimate of location / median still be valid

## Step 3 - Estimates of Variability
- dispersion: whether the data values are tightly clustered or spread out
- measure and reduce the variability
- distinguish random from real variability
- identify the various sources of real variability
- make decisions in the presence of it

### Deviations, Errors, Residuals
- The difference between the observed values and the estimate of location
- deviations from the mean = observed value - mean
- how dispersed the data is around the central value
-  #MeanAbsoluteDeviation: measure variability with the average of the absolute values of the deviations from the mean
-  #Variance: average of the squared deviations (preferred)
- #StandardDeviation: square root of the variance (preferred)
- working with squared values is much more convenient than absolute values
- all of them are not robust to outliers and extreme values

> Have n-1 in the variance formulas instead of n : concept of degrees of freedom. With n, we will underestimate the true value of the variance and the standard deviation in the population (biased estimate), with n-1 for an unbiased estimate

- #MedianAbsoluteDeviation: median absolute deviation from the median

### Percentiles 80 / Quantile 0.8
- look at the spread of the sorted data
- the Pth percentile is a value such that at least P percent of the values take on this value or less, and at least (100-P) percent of the values take on this value or more
- IQR (interquartile range)

## Step 4 - [Visualisation](Visualisation.md)

## Step 5 - 