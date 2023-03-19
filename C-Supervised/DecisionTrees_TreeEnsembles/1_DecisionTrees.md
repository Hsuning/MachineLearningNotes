#DecisionTree #TreeEnsembles
decision trees and tree ensembles

```toc
```

## Concept
- Non-parametric supervised learning algorithm for classification and regression, that predicts the value by learning simple decision rules (questions) inferred from the data features
- Example:
	- build a classifier to tell if a given animal is cat or not (1 or 0)
	- The features are ear shape, face shape, whiskers, which are categorical values (discrete values)
	- The root node is ear shape => is this animal with pointy or floppy ear shape?
![](Pasted%20image%2020230226162755.png)

## Learning Process
1. Start with all examples at the root node
2. Decide what feature to use by calculating #InformationGain + #EntropyFunction  all possible features, and pick the one with the highest information gain
3. Split dataset according to selected feature, and create left and right branches of the tree
4. Focus on the left branch, with steps from 2, and then right 
5. keep repeating splitting process until stopping criteria is met
	- when a node is 100% one class
	- when splitting a node will result in the tree exceeding a maximum depth (max_dept)
	- when improvements in purity score are below a threshold
	- when number of examples in a node is below a threshold
> One reason you might decide this is not worth sitting on is to keep the trees smaller and to avoid overfitting


###  Calculate entropy
-  #EntropyFunction is a measure of the impurity of a set of data. It starts from 0 goes up to 1 and then comes back down to zero as a function of the fraction of positive examples
- Select feature that **maximize purity (minimize impurity)**, means get towards subsets which are as close as possible to all cats or all dogs (only one category)
 ![](Pasted%20image%2020230311123005.png)
- Algorithms
	* Compute $p_1$, which is the fraction of examples that have value = `1` in `y`$$ \frac{sum(y=1)}{len(y)}$$
	* The entropy is then calculated as $$H(p_1) = -p_1 \text{log}_2(p_1) - (1- p_1) \text{log}_2(1- p_1)$$
	* If p1 = 0 or p_1 = 1, set the entropy to 0
	* make sure that the data at a node is not empty

### Split dataset
- takes in 
	- the data at a node
	- feature to split on
	- list of indices of data points
- Splits the indices of data points into left and right branches
	- If the value of `X` at that index for that feature is `1`, add the index to `left_indices`
	-  If the value of `X` at that index for that feature is `0`, add the index to `right_indices`
- We don't split data but indices to avoid duplications

### Information Gain
#InformationGain
- To reduce entropy / maximize entropy the most
$$\text{Information Gain} = H(p_1^\text{node})- (w^{\text{left}}H(p_1^\text{left}) + w^{\text{right}}H(p_1^\text{right}))$$
- $H(p_1^\text{node})$ is entropy at the node 
- $H(p_1^\text{left})$ and $H(p_1^\text{right})$ are the entropies at the left and the right branches resulting from the split
- take a weighted average: 
	- $w^{\text{left}}$ and $w^{\text{right}}$ are the proportion of examples at the left and right branch, respectively
	- combine entropy on all branches into a single number by taking a 
	- the node with a lot of examples with high entropy is worse than the node with a few examples with high entropy
$$
\frac{ \text{examples in this branche} }{ \text{total number of examples} } H(p1) + \frac{ \text{examples in this branche} }{ \text{total number of examples} } H(p0)
$$
- reduction in entropy is we hadn't split at all
	- maximum impurity is $H(0.5) = 1$ as we have 5 cats and 5 dogs
	- **stopping criteria for deciding when to stop splitting** is if the reduction in entropy is too small below threshold
	 - Algorithms
$$
H(0.5) - \frac{ \text{5} }{ \text{10} } H(0.8) + \frac{ \text{5} }{ \text{10} } H(0.2) = 0.28
$$
	- the reduction in entropy is 0.28
- calculate what the information gain would be from splitting on each of the features, then pick the one that gives you the maximum information gain

#RecursiveMethod  
The way we build the decision tree at root is by building other smaller decisions trees in the left and the right sub-branches.  

```
# Not graded
tree = []

def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth):
    """
    Build a tree using the recursive algorithm that split the dataset into 2 subgroups at each node.
    This function just prints the tree.
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
        branch_name (string):   Name of the branch. ['Root', 'Left', 'Right']
        max_depth (int):        Max depth of the resulting tree. 
        current_depth (int):    Current depth. Parameter used during recursive call.
   
    """ 

    # Maximum depth reached - stop splitting
    if current_depth == max_depth:
        formatting = " "*current_depth + "-"*current_depth
        print(formatting, "%s leaf node with indices" % branch_name, node_indices)
        return
   
    # Otherwise, get best split and split the data
    # Get the best feature and threshold at this node
    best_feature = get_best_split(X, y, node_indices) 
    tree.append((current_depth, branch_name, best_feature, node_indices))
    
    formatting = "-"*current_depth
    print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))
    
    # Split the dataset at the best feature
    left_indices, right_indices = split_dataset(X, node_indices, best_feature)
    
    # continue splitting the left and the right child. Increment current depth
    build_tree_recursive(X, y, left_indices, "Left", max_depth, current_depth+1)
    build_tree_recursive(X, y, right_indices, "Right", max_depth, current_depth+1)



  
build_tree_recursive(X_train, y_train, root_indices, "Root", max_depth=2, current_depth=0)

 Depth 0, Root: Split on feature: 2
- Depth 1, Left: Split on feature: 0
  -- Left leaf node with indices [0, 1, 4, 7]
  -- Right leaf node with indices [5]
- Depth 1, Right: Split on feature: 1
  -- Left leaf node with indices [8]
  -- Right leaf node with indices [2, 3, 6, 9]
```