# Definition
ML is about giving computers the ability to get better at some task by learning from data, instead of having to explicitly code rules.

A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience.

T = flag spam fro new emails
E = training data
P = need to be defined, e.g., ratio of correctly classified emails

# Four types of problems
- complex problems - no algorithmic solution
- replace long lists of hand-tuned rules
- build systems that adapt to fluctuating environments
- help humans learn (e.g., data mining).

# Model Parameter vs Hyperparameter
- A model has 1+ model parameters, they determine what it will predict given a new instance (e.g., the slope of a linear model). A learning algorithm tries to find optimal values for these parameters such that the model generalizes well to new instances. 
- A hyperparameter is a parameter of the learning algorithm itself, not of the model (e.g., the amount of regularization to apply).


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


## 2. Whether or not they can learn incrementally on the fly
### #OnlineLearning
- Can learn incrementally from a stream of incoming data
- Each learning step is fast and cheap
- Great for systems that receive data as a continuous flow (e.g., stock prices) and need to adapt to change rapidly or autonomously
- Only need limited computing r esources
- Can be used to train systems on huge datasets that cannot fit in one machine's main memory=> Out-of-core learning chops the data into mini-batches and uses online learning techniques to learn from these mini-batches
- cCapable of adapting rapidly to both changing data and autonomous systems
- Learning rate - how fast they should adapt to changing data: higher means learn rapidly, but also forget old data quickly / lower means less sensitvie to noise and outliers 
- The system's performance will gradually decline if bad data is fed to the system, need to monitor closely and promptly switch learning off and react to abnormal data
### #BatchLearning  = offline learning
- Cannot learn incrementally, model must be trained using all the available data, so offline
- Lot of time and computing resources
- Train a new version on the full dataset (old and new)
 
### 3. How they Generalize
Whether they work by simply comparing new data points to known data points, or instead by detecting patterns in the training data and building a predictive model, much like scientists do.
### #InstanceBasedLearning
The model learns the examples, then, when given a new case, it uses a similarity measure to find the most similar learned instances and uses them to make predictions.
### #ModelBasedLearning 
Tune some parameters to fit the model to the training set. Build a model and make predictions
*- Utility function: measures how good your model is*
*- Cost function: measures how bad it is, e.g., linear model to meaure the distance between the predictions and the examples, try to minimize this distance*
*- Attribute = data type, features = predictors = attribute + its value*

