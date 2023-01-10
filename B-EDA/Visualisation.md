```toc
```

## Data Distribution

### #Boxplot
- based on percentiles
- median by horizontal line in the box
- box represent 25th to 75th of data
- top and bottom of the box are the 75th and 25th
- dashed lines (whiskers): indicate the range for the bulk of the data

### #FrequencyTable
- a set of intervals (bins) and its count of data values

### #Histogram
- a plot of the frequency table with the bins on the x and the count on the y

### #DensityPlot
- a smoothed version of the histogram, often based on a kernel density estimate
- shows the distribution as a continuous line
- the y-axis is a proportion rather than count, the total area under the density curve is 1

## Binary and Categorical Data

### #Mode 
- the most frequent  values in the data

### #ExpectedValue
- a form of weighted mean
- the sum of values times their probability of occurrence
- it adds the ideas of future expectations and probability weights, often based on subjective judgment
- Ex: expected value of five years of profits from a new acquisition


### #BarCharts
- the frequency or proportion for each category plotted as bars

### #PieCharts
- the frequency or proportion for each category plotted as wedges in a pie

> We referred above to the probability of a value occurring


## Correlation
- examine correlation among predictors, and between predictors and a target variable
- Positively correlated: high values of X goes with high values of Y
- Negatively correlated: high values of X goes with low values of Y

### #CorrelationCoefficient
- Measures the extent to which numeric variables are associated with one another => always lies on the same scale
- Ranges from -1 to +1
- Pearson's correlation coefficient: multiply deviations from the mean for variable 1 times those for variable 2, and divide by the product of the standard deviations
- Sensitive to outliers
- Variables can have an no-linear association

### #CorrelationMatrix
- A table where the variables are shown on both rows and columns, and the cell values are the correlations between the variables

### #Scatterplot
- A plot in which the x-axis is the value of one variable, and the y-axis the value of another


## Multivariate analysis - exploring Two or More Variables

### #HexagonalBinning
- Numeric data only
- plot the relationship between 2 variables, rather than plotting points (without being overwhelmed by huge amounts of data)
- pd.plot.hexbin(): appear as a monolithic dark cloud, group the records into hexagonal bins and plotted the hexagons with a colour indicating the number of records in that bin
![](Pasted%20image%2020221214174313.png)
- sns.kedplot(): each contour band represents a specific density of points, increasing as one nears a peak
![](Pasted%20image%2020221214174328.png)

### #ContingencyTable
- Two categorical variables
- a table of counts by category

### #Boxplot 
- categorical and numeric data
- the distribution of a numeric variable grouped according to a categorical variable
- clearly shows the outliers in the data

### #ViolinPlot
- categorical and numeric data
- an enhancement to the boxplot
- plots the density estimate with the density on the y-axis
- shows nuances in the distribution that aren't perceptible in a boxplot

## Visualizing Multiple Variable
- plotting the data for a set of values
- use the arguments col and row to specify the conditioning variables
![](Pasted%20image%2020221214174934.png)