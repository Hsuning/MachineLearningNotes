#AnomalyDetection  #Unsupervised 

```toc
```

## Concept
- unlabelled dataset of normal events
- learns to detect or to raise a red flag is there is an unusual or an anomalous event
![](Pasted%20image%2020230415124128.png)

## Application
- Fraud detection: 
	- x_i = features of user i's activities
	- how often log in, how many web pages visited, transactions?, posts? , typing speed?)
	- model p(x) from data
	- identify unusual users by checking which have p(x) < e
	- perform additional checks to identify real fraud vs false alarms
- Manufacturing
	- x_i = features of product i
	- airplane engine, circuit board, smartphone
- Monitoring computers in a data center
	- x_i = features of machine i
	- based on memory use, number of disk accesses/sec, CPU load, CPU load/network traffics, ratio of CPU load to network traffics

## Density estimation
![](Pasted%20image%2020230415124702.png)
- Build a model for the probability of x
- What are the values that are less likely or have a lower chance / probability of being seen in the dataset
- For new test example X_test, 
	- compute the probability of being seen in the dataset p(x_test)
	- if it is small or less than some small number (called epsilon) $\epsilon$ means very unlikely to what we have seen = anomaly


### Gaussian (normal, bell-shape) distribution
![](Pasted%20image%2020230415142943.png)
- If the probability of `x` is given by a Gaussian distribution, with mean $\mu$, variance $\sigma^2$ 
- The probability of x looks like a curve, with center / middle of the curve $\mu$, standard deviation (width of this curve) $\sigma$ 
	- Standard deviation => $\sigma$
	- Variance of the distribution => $\sigma^2$
	-    $$ p(x ; \mu,\sigma ^2) = \frac{1}{\sqrt{2 \pi \sigma ^2}}\exp^{ - \frac{(x - \mu)^2}{2 \sigma ^2} }$$

- P(x):  if you get 100 numbers drawn from this probability distribution and plot a histogram with these number, the histogram looks vaguely bell-shaped
- Probabilities always have to sum up to 1, so the area under the curve is always equal to 1

### How to apply 
- given a dataset of m examples, **x is a vector**
- plot a histogram with these examples
- calculate $\mu$ (the average for each feature in the training examples) $$\mu_i = \frac{1}{m} \sum_{j=1}^m x_i^{(j)}$$
- and $\sigma^2$ = the variance for each feature in the training examples = average of the squared difference between training example and $\mu$ 
 $$\sigma_i^2 = \frac{1}{m} \sum_{j=1}^m (x_i^{(j)} - \mu_i)^2$$
 
```
mu = np.sum(X, axis=0) / m
var = np.sum((X-mu)**2, axis=0) / m
```

- These 2 formulas are technically called #MaximumLikelihoodEstimate for Mu and Sigma
- You can use $\frac{1}{m-1}$ but this makes very little difference

## Algorithm
![](Pasted%20image%2020230415145648.png)
- n features: 2 in example, but n can be much larger for many practical applications
- build the density estimation = a model that estimate the probability of `p(x)`
- ${\vec{x}^1, \vec{x}^2, ..., \vec{x}^m}$:  m training set
- $\vec{x}$ : a feature vector with values $x_1, x_2, x_n$ 
- $p(\vec{x})$: the product of probability of x1 `p(x1)`, times the probability of x2 `p(x2)`, ..., xN `p(xN)`
	- this equation assumes that the features are statistically independent, however, it works fine even that the features are not actually statistically independent.
	- $p(x_1) = p(x_1;\mu_1,\sigma_1^2)$ : probability of the individual feature by estimating the $\mu$ and $\sigma$ of the given **feature** x1
	- Why multiply probabilities: the chance for an engine to get an engine that both fronts really hot (1/10) and vibrates really hard  (1/20) = 1/200 = really unlikely


![](Pasted%20image%2020230415151522.png)
- $\mu_j$ : mu for feature j,  is the average of the feature $j$ of all the examples $x$
- $\sigma_k^2$: the average of the square difference between the feature and the value $\mu_j$
- With vectorised implementation, we can compute mu as the average of the training examples. $\vec{\mu}$ mu will be a vector
- Multiply p(x) together will tend to flag an example as anomalous if 1 or more features are either unusually large or unusually small related to what it has seen in the training set


## Real-number evaluation
- When developing a learning algorithms, making decisions is much easier if we have a way of evaluating the learning algorithm
- Build train, cv, test set
	- Assume we have some labeled data, of anomalous (y=1) and non-anomalous (y=0) examples
	- The training data is unlabelled, assume that all the training examples are normal (y=0)
	- Create cross validation set and test set that includes a few number of anomalous (y=1) examples and a lot of normal example (y=0)
	- It is okay to have some examples that are actually anomalous but were accidentally labeled with y=0
	- Alternative No test set if only very few labeled anomalous examples, with higher risk of overfitting
![](Pasted%20image%2020230415161202.png)

- Step
	- Fit model p(x) on training set
	- On a cross validation / test example x, predict 1 if p(x) < e and 0 if p(x) > e
	- Possible evaluation metrics
		- True positive, false positive, false negative, true negative
		- Precision / Recall
		- F1-Score
	- Use cross validation set to look at how many anomalies is finding and how many normal engines is incorrectly flagging as an anomaly

### Selecting the threshold $\epsilon$
- the low probability examples are more likely to be anomalies
- to determine which examples are anomalies, one way is to select a threshold using F1 score based on a cross validation set
- Algorithms
	- a loop that will try many different values of $\epsilon$
	- calculate the tp, fp, fn, precision, recall and then F1 score
	- compare with best_F1 and replace if the score is better
```
def select_threshold(y_val, p_val): 
    """
    Finds the best threshold to use for selecting outliers 
    based on the results from a validation set (p_val) 
    and the ground truth (y_val)
    
    Args:
        y_val (ndarray): Ground truth on validation set
        p_val (ndarray): Results on validation set
        
    Returns:
        epsilon (float): Threshold chosen 
        F1 (float):      F1 score by choosing epsilon as threshold
    """ 

    best_epsilon = 0
    best_F1 = 0
    F1 = 0
    
    step_size = (max(p_val) - min(p_val)) / 1000
    
    for epsilon in np.arange(min(p_val), max(p_val), step_size):
    
        ### START CODE HERE ### 
        y_pred = p_val < epsilon
        
        tp = sum((y_pred == 1) & (y_val == 1))
        fp = sum((y_pred == 1) & (y_val == 0))
        fn = sum((y_pred == 0) & (y_val == 1))
        
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        F1 = (2 * precision * recall) / (precision + recall)
        
        
        ### END CODE HERE ### 
        
        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
        
    return best_epsilon, best_F1
```


## Anomaly detection vs Supervised learning
#### Anomaly
- very small number of positive examples (y=1). (0-20 is common)
- Large number of negative (y=0) examples
- Many different "type" of anomalies. Hard for any algorithm to learn from positive examples what the anomalies look like
- Future anomalies may look nothing like any of the anomalous examples we've seen so far (completely new)
- Examples:
	- Fraud detection
	- Manufacturing - finding new previously unseen defects
	- Security related application such as monitoring machines in a data center (hacked)

#### Supervised
- Large number of positive and negative examples
- Enough positive examples for algorithm to get a sense of what positive examples are like
- Future positive examples likely to be similar to ones in training set
- Example:
	- Email spam
	- Manufacturing - finding known, previously seen defects
	- Weather prediction (sunny / rainy / etc.)
	- Diseases classification


### Choosing what feature to use
- **more important** for unsupervised learning, as it runs just from unlabelled data and is harder to figure out what features to ignore
- it might be okay for supervised learning, with label the algorithm can figure out what features to ignore or how to scale feature

#### Non-gaussian features
- Ensure the features are more or less Gaussian
- Transform non-gaussian features in order to make more Gaussian
- When anomaly detection models P(x) using a Gaussian distribution, it is more likely to be a good fit to the data
```
x1 = np.log(x1) 
# might get error is you have value in x equal to 0

x2 = np.log(x2 + 1) 
# np.log(x2 + c) to avoid error with 0

x3 = x3 ** 0.5 
# x3 ** power
```
- a larger value of C will end up transforming this distribution less
- try different power and visualise with histograms

#### Error analysis for Anomaly detection
- The target 
	- $P(x) >= \epsilon$ to be large for normal examples $x$
	- $P(x) < \epsilon$ to be small for anomalous examples $x$
- Problem
	- $P(x)$ is large for both normal and anomalous examples
	- The anomalous examples look similar to normal examples
- Solution
	- Choose features that might take on unusually large or small values in the event of an anomaly
		- memory use of computer, number of disk accesses/sec, CPU load, network traffic
	- Train the model
	- See what anomalies in the cross-validation set the algorithm is failing to detect
		- High CPU load and low network traffic
	- Look at that example and see if that can inspire the creation of new features that help distinguish this example form the normal example
		- new feature = CPU load / network traffic
		- new feature = (CPU load) ** 2 / network traffic