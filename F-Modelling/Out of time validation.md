#In-TimeValidate #In-TimeTest #Out-of-TimeValidate #Out-Of-TimeTest

We have data from 2015 to 2018.
We want to predict 2019.

![](Pasted%20image%2020230502114205.png)

## In-time validate, in-time test
- train, validate and test the model using data from the **same time period**
- make prediction for new data *2019* using a model trained, validated. and tested on labeled *2015 - 2018* data
- Pro:
	- use the most recent data to train the model
- Con:
	- it assumes that the relationships that existed in the past will be the same in the future, if the relationships shift over time, model will perform worse than we expect
	- no idea how model performs out-of-time

## In-time validate, out-of-time test
- hold-out 2018
- Train the model using *2015-2017* data, and test it on *2018*
- Pro:
	- evaluate how the model performs out-of-time
- Con:
	- performance out-of-time not taken into account during model building
	- not using the most recent data to make predictions unless we re-train the model using the full data

## Walk forward / Sliding windows
#Out-of-TimeValidate 
- each time we retrain the model using all observations through a particular point in time, and make predictions for the next time period
- train many different models and choose one that creates best combined predictions
- Pro:
	- model building takes into account ability to predict out-of-time
	- the model is refit for each window, this approach simulates periodic retraining of the model which happens in practice
- Con:
	- always validate predictions on the same set of data rather than randomly selected data
	- perhaps holding out a portion of the test data could overcome this concern
