# Type of Machine Learning

```toc
```

## Supervised
#Supervised

### Concept
- Learn From Being Given "right answers"
- Data = Labelled data with desired solution (label) for each instance
- By seeing correct pairs of input x and desired output label y, learns to take just the input alone without the output label and gives a reasonably accurate prediction of the output
- Given a training set, to learn a function $h:X -> y$ so that h(x) is a good predictor for the corresponding value of y
- Use Cases: span filtering, speech recognition (multi-class classification), machine translation, online advertising, self-driving car, visual inspection

### Regression
#Regression
- Predict a number, infinitely many possible outputs
- Target variable is continuous $(3399, 3943.23, 39432)$

### Classification
#Classification
- Predict categories, small number of possible output
- Target variable is a discrete value $(1, 0)$ or even no-numeric ($cat, dog$)

### Algorithms
- k-Nearest Neighbors
- Linear Regression (only for regression, it can be used for classification, but you won't be able to get the best result)
- Logistic Regression (only for classification)
- Support Vector Machines (SVMs)
- Decision Trees and Random Forests
- Neural networks

> Some neural network architectures can be unsupervised, such as autoencoders and restricted Boltzmann machines. They can also be semi-supervised, such as in deep belief networks and unsupervised pretraining.

## Unsupervised 
#Unsupervised

### Concept
- Find something interesting in unlabelled data
- Only comes with inputs $x$
- Automatically find **structure** in the data and automatically figure out what the major types of individuals (data points)

### Clustering
#Clusting
- Detect groups of similar data points.
- Algorithms
	- K-Means  
	- DBSCAN
	- Hierarchical Cluster Analysis (HCA)

### Anomaly Detection & Novelty Detection
- https://scikit-learn.org/stable/modules/outlier_detection.html
- Decide whether a new observation belongs to the same distribution as existing observations (it is an _inlier_), or should be considered as different (it is an _outlier_)
- Both used for anomaly detection
- Outliers = observations that are far from the others  

| Definition    | Anomaly #AnomalyDetection                                                                                            | Novelty  #NoveltyDetection                                                 |
| ------------- | --------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| Training data | Contains outliers                                                                                   | Not polluted by outliers                                 |
| Use           | Fit the regions where the training data is the most concentrated, ignoring the deviant observations |  |
| Goal          | Detect abnormal or unusual observation                                                              | Detect whether a new observation is an outlier (novelty) |
| Nature        | Unsupervised                                                                                        | Semi-Supervised                                          |
| Visualisation | Cannot form a dense cluster as available estimators assume that the outliers/anomalies are located in low density regions | Can form a dense cluster as long as they are in a low density region of the training data, considered as normal in this context |

- Algorithms:
	- Isolation Forest
	- Local Outlier Factor
	- One-class SVM _requires fine-tuning of hyperparameter nu_
	- SGD Once-class SVM: linear one-class SVM with linear complexity in the number of samples
	- Elliptic Envelope: Guassian data, learns an ellipse

	> If the dataset is imbalanced, like pictures of fruits and only 1% represent bananas, novelty should not consider new pictures of bananas as novelties, but anomaly may consider these pictures as so rare and so different from other and classify them as anomalies.*

### Dimensionality Reduction
 #DimensionalityReduction
 - Simplify the data without losing too much information
 - Merging several correlated features into one, output a 2D or 3D representation of complex and unlabelled data that can easily be plotted
 - Use to understand how the data is organised and perhaps identify unsuspected patterns
 - Model can run faster, and the data will take up less memory space
 - Ex: car's mileage may be strongly correlated with its age, merge them into one feature that represents the car's wear and tear. Compress data using fewer numbers.
 - Algorithms:
	- Principal Component Analysis (PCA)
	- Kernel PCA  
	- Locally Linear Embedding (LLE)
	- t-Distributed Stochastic Neighbour Embedding (t-SNE)

### Association Rule Learning
#AssociationRuleLearning
- Dig into large amounts of data and discover interesting relations between attributes.
- Ex: people who purchase barbecue sauce and potato chips also tend to buy steak.
- Algorithms:
	- Apriori
	- Eclat

## Semi-supervised
#Semisupervised
- Deal with data that's partially labelled.  
- Ex: family photos,  
	1. Clustering to identify that the person A is in photos 1, 5 and 11  
	2. Add few labels per person and the model will be able to name everyone in every photo  
	3. Manually clean up some clusters.
- Most semi-supervised learning algorithms are combinations of unsupervised and supervised algorithms.
- Deep belief networks (DBNs): based on unsupervised components called restricted Boltzmann machines (RBMs). Train RBMs sequentially and fine-tune using supervised learning techniques.

## Reinforcement Learning
#ReinforcementLearning  
- When we want a robot to learn to walk in various unknown terrains
- Learning system (_agent_) observe the environment, select and perform actions, and get positive/negative _rewards_ in return. Then learn by itself what is the best strategy to choose when it is in a given situation (_policy_).
-   Reinforcement learning: This type of machine learning involves training a model to make decisions based on rewards and punishments. The goal of reinforcement learning is to find the optimal strategy for a given task. Examples of reinforcement learning algorithms include Q-learning and Monte Carlo Tree Search.


## Transfer Learning
- Use a model that has been trained on one task as the starting point
- Leverage the knowledge gained from the pre-trained model
- To train a new model on a different but related task


## Deep learning
- Artificial neural networks - composed of many interconnected processing nodes
- automatically learn complex, non-linear relationships in data, and they have been used to achieve state-of-the-art performance in a variety of tasks, such as image and speech recognition.