# 3_Evaluation Metrics
```toc
```

## Establishing a Baseline Level of Performance
- Establish a baseline level of performance through learning algorithm will make it much easier for us to look at theses loss and judge if they're high or low
- Human level performance: see if the training error is much higher than a human level of performance. Especially good for audio images or text

```
Speech recognition example
Human level performance: 10.6%
Training error: 10.8%
Cross validation error: 14.8%
```

	- The training error means a high bias, however, when we benchmark it to human level performance, we see that the algorithm is actually doing well
	- However, the cross validation error is much higher than the training error, the algorithm actually has more of a variance problem
- Competing algorithms performance: based on a previous implementation, or even a competitor's algorithm
- Guess based on experience
- 2 key quantities to measure:
	- Bias: gap between training error and baseline performance
	- Variance: gap between training error and cross-validation error


#EvaluationMetrics #CostFunction

## Evaluation Metrics / Cost Function
- Skewed dataset: The ratio of pos and neg are very different. #Accuracy don't work that well. For example, only 0.5% of patients have the disease, simply predicting all the patients don't have disease will get 99.5% accuracy, only 0.5% error  
![](Pasted%20image%2020230108165248.png)
- #ConfusionMetric
	- #TruePositive #TrueNegative: the predicted value is correct
	- #FalsePositive #FalsePositive: the predicted value is incorrect
-  #Precision
	- for all patients where we predicted $y=1$, what fraction do we actually get right?
	- $\frac{True\_positives}{Predicted\_Positive = True\_pos+False\_pos}$
- #Recall
	- of all patients that actually have the rare disease, what fraction did we correctly detect as having it?
	- $\frac{True\_positives}{Actual\_Positive = True\_pos+False\_neg}$

### Precision Vs Recall
- Precision: can algorithm provide an accurate diagnostic
- Recall: can algorithm correctly identify all the patients with rare disease  
![](Pasted%20image%2020230108170757.png)
- Predict y=1 (rare disease) **only if very confident**
	- a disease means a possibly invasive and expensive treatment, the consequences of the disease aren't that bad even if left not treated aggressively
	- set a higher threshold (y=1 only when the probability is > 0.7 )
	- Higher precision and lower recall:
- Predict y=1 when in doubt to **avoid missing too many cases of rare disease**
	- leaving a disease untreated has much worse consequences to the patient
	- set a lower threshold
	- Lower precision and higher recall
- Plotting allow us to get a point we want. We might need to manually select the point
- #F1Score
	- a way to combine Precision and Recall into single score, *gives more emphasis to whichever of these values is lower* (as an algorithm with very low precision/recall might not be useful)
	- Harmonic mean of P and R: a way of taking an average that emphasises the smaller values more
	- $2*\frac{Precision*Recall}{Precision+Recall}$
