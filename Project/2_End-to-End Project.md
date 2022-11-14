# Introduction
Main steps for analytics machine learning project.

## 1.  Look at the big picture.
d

## 2.  Get the data.
    
## 3.  Discover and visualize the data to gain insights.
    
## 4.  Prepare the data for Machine Learning algorithms.
    
## 5.  Select a model and train it.
    
## 6.  Fine-tune your model.
    
## 7.  Present your solution.
    
## 8.  Launch, monitor, and maintain your system.


In my point of view, the first step of the machine learning project is always to understand the business needs and frame the problem with end-users. End-users can be a business team or even another data science team. Building a model is important, but not the end goal. The goal would be how does the company and the end-users expect to use and benefit from the model. Further, it is also important to understand what is the current solution, what will be changed with the new solution.  

  

Once we have all the information, we can start the solution design (how to get data, where to build model, how to deploy, task - supervised, unsupervised or ?, learning method, etc.). At this stage, we shall also select some performance measures, both technical and business, that can make end-users feel comfortable and can be used to maintain the model performance. At the end, we shall check all the assumptions with business to verify that the design fulfills all the needs.

  

The next stage would be to get the data, build an ETL pipeline if needed. A data quality assessment and a data dictionary are also important, as sometimes we might apply some filters on the data, or sometimes the data set comes from multiple sources. This allows us to observe the data trend (for example, after deployment, we might need to retrain the model when there is any huge change in the data). Moreover, we can get some tastes of the data.

  

Once we confirm the data that we want to use for training, we then prepare the data for machine learning algorithms and do some exploratory data analysis to gain insights with some visualisations. Based on the task predefined, we will use different exploration methods. For instance, we look inside the target variables and their correlations compared to other independent variables for a supervised task, but look inside the occurrence / frequency of words, most common words, etc., for a text clustering task (unsupervised). The data preparation methods differ from the task too, but generally covers data transformation, outliers preprocessing, one-hot encoding and vectorization, feature engineering, etc.  

  

After all, we get the training set and start to select the algorithmes and training. In general, we split the dataset into training (with validation dataset) and test set. The test set will not be touched until we choose the final model and confirme the hyperparameters we want to use after fine-tuning. At this stage, we also verify the feature we created previously, different preprocessing methods, and refine the training dataset. Once we choose the model (probably from 1 to 3), we then refine the hyperparameters. A typical way is to split the training set using cross validation techniques, for example, we can use cross-validation to improve the transformer model validation for BERT text-classification model. Here, the model and hyperparameters selection shall be based on technical and business evaluation metrics. Possibly, we will combine the models that perform best using ensemble methods, which often perform better than the best individual model. We shall also analyze the best models and their errors, to see whether there can be any further improvement. Finally, we evaluate the final model on the test set to estimate the generalisation of our model. If there is any generalization error, we might need to reselect and fine-tune the models.  

  

If everything goes well, we will be at the project pre launch phase. Here, we shall prepare some easy-to-understand presentations (visualizations, statements, insights) to introduce the whole solution, document everything, prepare for production (version control, documentation and tests, maintenance notes), define the monitoring methods (dashboard, daily report, KPIs report for business, and codes with some performance check and trigger alerts for developers) and maintenance. It is important to prepare an automated pipeline that collects fresh data regularly, trains the model, fine-tunes the hyperparameters and compares with previous models.