#Clustering #Unsupervised #KMeans 

## Concept
- a method to automatically cluster similar data points together: 
	1. Given a training set
	2. K-means starts by taking a random guess at where might be the center (called cluster centroids)
	3. Refines this guess by repeatedly 
		- assigning each point to their closest cluster centroid
		- take an average of all the data points and move the cluster centroid to the average location (mean) for each clusters
	1. Until converge = no more changes to the cluster of the points or to the locations of the cluster centroids

## Algorithms
![](Pasted%20image%2020230407115436.png)
- k : the number of clusters
- $\mu$ : average of points assigned to cluster k, they are vectors with same dimensions as the training examples 
- m: data points / training examples
- $x^{(1)}$: ith training examples
- $x^{(1)}, c^{(2)} = 2$: the cluster centroid closest to $x^{(1)}$ is 2
- $min_k || x^{(1)} - \mu_k || ^2$:  L2 norm, the distance between two points
- The inner-loop of the algorithm repeatedly carries out 2 steps:
	- assigning each training example $x^{(i)}$ to its closest centroid
	- and recomputing the mean of each centroid using the points assigned to it on the horizontal axis and then vertical axis
	![](Pasted%20image%2020230407114936.png)
	

## Convergence
The k-means will always converge to some final set of means for the centroids
Improvement. However, the solution may not always be ideal :
- on corner case : a cluster has zero training examples assigned to it
- frequently applied to clusters that are not well separated
	- how to size S, M, L t-shirt
	- collect data based on height and weight of customers

Hence, it is recommend to
- run a few times with different random initializations
- eliminate or increase the number of k
- choose the one with the lowest cost function value (distortion)


## Implementation 1: Find closest centroids
- training data = a matrix `X`
- total number of centroids = `K`
- centroids = `centroids`
- output = one-dimensional array `idx` 
	- with same number of elements as X
	- with the index of the closest centroid from 0 to k-1
- Algorithms 
	- $$c^{(i)} := j \quad \mathrm{that \; minimizes} \quad ||x^{(i)} - \mu_j||^2,$$
	- $c^{(i)}$ : the `idx` of the centroid that is closest to x = `idx[i]` 
	- $\mu_j$ is the position (value) of the $j$’th centroid = stored in `centroids` in the starter code)
```
def find_closest_centroids(X, centroids):

    # Set K
    K = centroids.shape[0]

    # Output
    idx = np.zeros(X.shape[0], dtype=int)

	# For every element in training dataset
    for i in range(X.shape(0)):
	    # Distance between X[i] and each centroids[j]
        distance = []
        
        for j in range(K):
	        # Calculate the norm
            norm_ij = np.linalg.norm(X[i] - centroids[j])
            distance.append(norm_ij)
            
		# Calculate index of minimum value
        idx[i] = np.argmin(distance)
        
    return idx
```

## Implementation 2 : compute_centroids
- recompute the value for each centroid
* Specifically, for every centroid $\mu_k$ we set
	* $$\mu_k = \frac{1}{|C_k|} \sum_{i \in C_k} x^{(i)}$$
	* $C_k$ = the set of examples that are assigned to centroid $k$
	* $|C_k|$ is the number of examples in the set $C_k$
- If two examples say $x^{(3)}$ and $x^{(5)}$ are assigned to centroid $k=2$, update $\mu_2 = \frac{1}{2}(x^{(3)}+x^{(5)})$.

```
def compute_centroids(X, idx, K):
    
    m, n = X.shape
    
    centroids = np.zeros((K, n))
    
    for i in range(K):
	    points = X[idx == i]
        centroids[i] = np.mean(points, axis=0)
        
    return centroids
```

## Implementation Final - Run K-means
```
def run_kMeans(X, initial_centroids, max_iters=10, plot_progress=False):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example
    """
    
    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids    
    idx = np.zeros(m)
    plt.figure(figsize=(8, 6))

    # Run K-Means
    for i in range(max_iters):
        
        #Output progress
        print("K-Means iteration %d/%d" % (i, max_iters-1))
        
        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)
        
        # Optionally plot progress
        if plot_progress:
            plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
            
        # Given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K)
    plt.show() 
    return centroids, idx
```


## Cost Function / Optimization objective
#DistortionCostFunction #CostFunction 
![](Pasted%20image%2020230410115939.png)
- The dimension of $\mu_k$​ matches the dimension of the examples => if each example x is a vector of 5 numbers, then each cluster centroid is also going to be a vector of 5 numbers
- $c^{(i)}$ describes which centroid example (i) is assigned to, so the number is equal to the number of training examples
- The algorithm for cluster is trying to optimize the cost function
- Minimize j by assign $x^{(i)}$ to the closer cluster centroid without chaining $\mu$ $$c^{(i)} := \text{index of cluster centroid closest to } x^{(i)} $$
- Minimize j by choosing the term $\mu_{c^{(i)}}$ 
	- Origin $\mu^k$ = $\frac{1}{2} (1^2 + 9^2) = 41$
	- After moving to the average place, $\mu^k$ = $\frac{1}{2} (5^2 + 5^2) = 25$
	- Get a much smaller average square distance 
- On every integration, the distortion cost function should go down or stay the same
	- It should never go up, otherwise there is a bug
	- stay the same or go down very very slow means the k-mean has converge or closer to convergence so stop running


## How to initalizing k-means
- The number of cluster central **k** shall be less than the number of training examples **m** $$\text{choose } k < m $$
- A good strategy for initializing the centroids is to randomly pick K training examples 
	- Randomly shuffles the indices of the example `np.random.permutation()`
	- Select the first `K` examples
	- Set $\mu_{1}, \mu_{2}, ..., \mu_{k}$ equal to these K examples
- To avoid stuck in a local minimum, run multiple times to try to find the best local optima.
- To choose between these solutions, compute the cost function J for all clusters found by k means, pick the one that gives you the lowest value for the cost function j.

```
def kMeans_init_centroids(X, K):
    """
    This function initializes K centroids that are to be 
    used in K-Means on the dataset X
    
    Args:
        X (ndarray): Data points 
        K (int):     number of centroids/clusters
    
    Returns:
        centroids (ndarray): Initialized centroids
    """
    
    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])
    
    # Take the first K examples as centroids
    centroids = X[randidx[:K]]
    
    return centroids
```

Random initialization
![](Pasted%20image%2020230414111422.png)
- Run between 50 - 1000 times. 
	- At least 50 to 100 random initializations will often give you a much better result
	- Not over 1000 as It can be computational expensive




## Choosing the number of clusters
- the right value of k is truly ambiguous

### Elbow method
- run k-means with a variety of values of K and plot the cost function / distortion function J  
- hardly use this methods as the right k is often ambiguous and a lot of cost functions will just decrease smoothly and doesn't have a clear elbow by wish 
- **Don't choose K just to minimize cost J**
![](Pasted%20image%2020230414112310.png)

Choosing the value of K
- Purpose : evaluate k-means based on how well it performs on your purpose
