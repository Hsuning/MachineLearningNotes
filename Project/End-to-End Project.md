```python
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight(
    class_weight='balanced',
    y=train_df['class'] #provide your own target name
)
```

Main steps for analytics machine learning project.

## Look at the big picture / Problem Statement
- Purpose: 
	- Better understand what drives claims, who is more likely to claim, and by how much
	- Build a model to identify better/worse risks when it comes to providing car insurance
	- Accuracy of the solution, based the metrics I defined
	- Any non-technical considerations hat can be drawn out
	- prevent adverse selection, predict and prevent
	- give our customers more information about the risks that they are taking and how they can themselves adopt a more proactive attitude to risk.
- Data: drivers information, claims history

## Get the data
The dataset contains drivers' information and their claims history. 
The data includes metrics such as
- ID (not unique): REFERENCE
- demographic: BIRTHDAY, AGE, EDUCATION, HOME_CHILDREN, HOME_VAL, INCOME, MARITAL_STATUS, Gender, (OCCUPATION, JOB_TENURE)
- who in the car: CHILD_DRIVE, MARITAL_STATUS
- make, model and age of car: CAR_COST, CAR_AGE, CAR_TYPE
- overall car usage: CAR_USE, (OCCUPATION, JOB_TENURE)
- any claims and convictions: CLM_FREQ, MVR_PTS, OLDCLAIM, REVOKED, CLAIM_FLAG
- PTIF
- frequency/length of driving time: TRAVTIME
- traffic conditions: CITY_RURAL
- Target variable: CLM_AMT

There wasn't any information related to date of accident or claims. However, the dataset has driver's age.

- REFERENCE: ID is not unique. Need 


## Discover and visualize the data to gain insights
	- View the variables
	- Check the dimensions of variables
	- Visualize data
    
## Prepare the data for Machine Learning algorithms.
    
## Select a model and train it.
    
## Fine-tune your model.
    
## Present your solution.
A report briefly describing the methodology
pros and cons of the solution
any considerations
reference to any potential alternative solutions that were considered

## Launch, monitor, and maintain your system.

