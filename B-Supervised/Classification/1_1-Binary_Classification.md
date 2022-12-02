# Binary Classification
#BinaryClassification

## Concept
- Classification problems:
	- identifying email as Spam or Not Spam
	- Determining if a tumour is malignant or benign
- Target variable only have two possible values (class, category)
- Very commonly using the number to represent class
	- 0: False, No, Negative, Absence (not always bad)
	- 1: True, Yes, Positive, Presence (not always good)
- Why can't we use #LinearRegression ?
	- It predicts all numbers between 0 and 1, and even less than 0 or greater than 1
	- It does not match the data well. One option to improve the result is to apply a **threshold**
	- **However**, adding more example to the right would change our conclusions about how to classify the data
	- End up with learning a much worse function  
![[Pasted image 20221118103209.png]]

## Decision Boundary
- Dividing line of two categories
