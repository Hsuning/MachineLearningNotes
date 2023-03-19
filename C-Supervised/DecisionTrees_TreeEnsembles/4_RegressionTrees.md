## Concept
- Regression with Decision Trees
- Same algorithms as #DecisionTree , but the target is based on the average of all the values in the training samples for each branches

![](Pasted%20image%2020230314214937.png)
- given a new test example
	- follow the decision nodes down as usual until it gets to a leave node
	- predict the value at the leaf node
- choosing a split
	- split on a given feature, end up with samples on the left and right with the corresponding target values
	- reduce the variance (how widely a set of numbers varies) of the weights of the values Y at each of these subsets
	- H(p) = variance
	- #InformationGain  reduction in variance
$$\text{Information Gain} = H(p_1^\text{node})- (w^{\text{left}}H(p_1^\text{left}) + w^{\text{right}}H(p_1^\text{right}))$$