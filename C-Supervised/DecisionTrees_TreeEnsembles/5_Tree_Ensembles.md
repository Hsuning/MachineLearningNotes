#TreeEnsembles 



## Concept
- #DecisionTree can be highly sensitive to small changes in the data. Changing just one training example causes the algorithm to come up with a different split at the root and a totally different tree
- To make the algorithm less sensitive and more robust, we can build a lot of decision trees, get them to vote on what is the final prediction, use the majority vote of the predictions

## Sampling with replacement
#SamplingWithReplacement
- Concept
	- without replace, we will always just get the same dataset
	- before the selection, put all samples back in the data and shuffle them again
	- then get same number of samples but with replacement
	- ensures that we don't just get the same data
- Benefit
	- build multiple random training sets that are all slightly different from the original one, possibly don't contain all the training examples
	- explore a lot of small changes in the data and train different decision trees
	- averaging over a lot of small changes to the training set before wrapping up
	- any little change further to the training set makes it less likely to have a huge impact on the overall output of the overall algorithm

## Random feature selection
#RandomFeatureSelection
- Concept
	- At each node, pick a random subset of  k < n - total number of features, and allow the algorithm to only choose from that subset of features
- Benefit
	- even with this sampling with replacement procedure, sometimes we end up with always using the same split at the root node and very similar splits near the roots node
	- cause the set of trees learned to be more different from each other, so more accurate prediction
- how to choose k
	- square root of n

## Random forest
#RandomForest
- Given training set of size m
- For $b$ in $B$
	- Use #SamplingWithReplacement to create a new training set of size $m$
	- training examples are different every times, some samples are repeated
	- #RandomFeatureSelection to create a new list of features
	- Train a decision tree on the new dataset and random features
	- Get these trees to vote on the correct final prediction
- $B$ refers to the number of trees we want to build
	-  #BaggedDecisionTree: put training examples in a virtual bag, so use letters lowercase b and uppercase B here = bags
	- the recommended value is from 64 to 128
	- having many trees never hurts performance, but beyond certain point will end up with diminishing returns, so the performance wouldn't be better when B is > 100
	- Never use 1000 trees as will slow down the computation significantly


## Boosted decision tree
#XGBoost #BoostedDecisionTree
- Given training set of size m
- For b in B:
	- Use #SamplingWithReplacement to create a new training set of size m
	- Instead of picking examples with equal (1/m) probability, make it more likely to **pick examples that previously trained trees misclassify**
	- look at what the ensemble of trees (1, 2, .., b-1) are not doing that well on
	- higher probability of picking examples that the ensemble of the previously built trees is still not yet doing well
	- Train a decision tree on the new dataset
- Deliberate practice
	- rather than practice all pieces
	- play the piece that you aren't yet playing that well

### XGBoost (eXtreme Gradient Boosting)
#XGBoost 
- Open source
- Fast efficient implementation
- Good choice of default splitting criteria and criteria for when to stop splitting
- built in regularisation to prevent overfitting
- highly competitive algorithm for machine learning competitions
- Algorithm:
	- Similar to boosted decision tree
	- rather than #SamplingWithReplacement , it assigns different weights to different training examples, so no need to generate a lot of randomly chosen training set (more efficient


## Decision Trees vs Neural Network
#DecisionTree 
- tabular (structured) data
- not recommended for unstructured data (images, audio, text)
- fast to train
- small decision trees may be human interpretable

#NeuralNetwork 
- Works well on all types of data, including tabular and unstructured data
- Slower than a decision tree
- Works with transfer learning
- When building a system of multiple models working together, it might be easier to string together multiple neural networks
