# 1_Development Process
```toc
```

## Full Cycle of a Machine Learning Project
![](Pasted%20image%2020230108121444.png)

### Scope Project
- what is the project
- what you want to work on
- Ex: I want to work on speech recognition for voice search that is to do web search using speaking to your mobile phone rather than typing into your mobile phone

### Collect Data
- decide the data needed to train the model
- Ex: get the audio, transcripts or labels

### Train Model
- Training, #ErrorAnalysis , #IterativeImprovement
 - Might want to go back to collect more data of every thing or specific type
 - Ex: model performs poorly when there is a car noise, so collect more data (#DataAugmentation) to get more speech data that sounds like it was a car

### Deploy in Production
- Deploy, monitor and maintain system
- Available for users to use
- Not working as expect, so go back to training or even collect data

**Deployment**
- Take the model and implement it in a server called **Inference Server**, call the model in order to make prediction
- Mobile app with search application, make API call to pass audio clip that was recorded
- Inference server apply the ML model and return the prediction in text transcript
- Software engineering:
	- **Ensure reliable and efficient predictions**
	- **Scaling** to a large number of users
	- **Logging**: log the data (x and y-hat), assuming that user privacy and consent allows you to store this data
	- System monitoring: figure out when the data was shifting and the algorithm was becoming less accurate
	- Model updates
- #MLOps: machine learning operations, the practice of how to systematically build and deploy and maintain machine learning systems to do all of these things

## Iterative Loop of ML development
#IterativeImprovement
- Choose architecture: model, data, etc.
- Implement and train a model
- Diagnostics: bias, variance, error analysis
- Based on the insight, improve architecture by adding more data / features, subtract features
