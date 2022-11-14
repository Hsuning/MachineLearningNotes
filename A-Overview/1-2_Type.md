# Type
## 1. Whether or not they are trained with human supervision
### #Supervised 
Labeled data with desired solution (label) for each instance. Classification (labels, discrete values), regression (target, numeric value)
- k-Nearest Neighbors
- Linear Regression
- Logistic Regression
- Support Vector Machines (SVMs)
- Decision Trees and Random Forests
- Neural networks
	*Some neural network architectures can be unsupervised, such as autoencoders and restricted Boltzmann machines. They can also be semisupervised, such as in deep belief networks and unsupervised pretraining.*

### #Unsupervised 
Unlabeled data
- #Clustering : detect groups of similar people/visitors
	- K-Means  
	- DBSCAN
	- Hierarchical Cluster Analysis (HCA)
- #AnomalyDetection - detect unusual data or outlier by learning how to recognize normal instances during training;  
  #NoveltyDetection - detect new instances that look different from all instances in the very clean training set, without the instance that we want the algorithm to detect. 
  *If the dataset is imbalanced, like pictures of fruits and only 1% represent bananas, novelty should not consider new pictures of bananas as novelties but anomaly may consider these pictures as so rare and so different from other and classify them as anomalies.*
	- One-class SVM  
	- Isolation Forest  
- #Visualization #DimensionalityReduction : simplify the data without losing too much information by merging serveral correlated features into one, output a 2D or 3D representation of complex and unlabeled data that can easily be plotted, so we can understand how the data is organized and perhaps identify unsuspected patterns. Ex: car's mileage may be strongly correlated with its age, merge them into one feature that represents the car's wear and tear. 
	- Principal Component Analysis (PCA) 
	- Kernel PCA  
	- Locally Linear Embedding (LLE)
	- t-Distributed Stochastic Neighbor Embedding (t-SNE)
*Model can run faster and the data will take up less memory space.*
- #AssociationRuleLearning : dig into large amounts of data and discover interesting relations between attributes. Ex: people who purchase barbecue sauce and potato chips also tend to buy steak.
	- Apriori 
	- Eclat

### #Semisupervised 
Deal with data that's partially labeled. 
Ex: family photos, 
	1) Clustering to identify that the person A is in photos 1, 5 and 11
	2) Add few labels per person and the model will be able to name everyone in every photo
	3) Manually clean up some clusters.
- Most semisupervised learning algorithms are combinations of unsupervised and supervised algorithms.
- Deep belief networks (DBNs): based on unsupervised components called restricted Boltzmann machinees (RBMs). Train RBMs sequentially and fine-tune using supervised learning techniques.

### #ReinforcementLearning 
When we want a robot to learn to walk in various unknown terrains:
Learning system (*agent*) observe the environment, select and perform actions, and get positive/negative *rewards* in return. Then learn by itself what is the best strategy to choose when it is in a given situtaion (*policy*).