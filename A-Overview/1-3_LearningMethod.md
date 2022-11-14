
## Whether or not they can learn incrementally on the fly
### #OnlineLearning
- Can learn incrementally from a stream of incoming data
- Each learning step is fast and cheap
- Great for systems that receive data as a continuous flow (e.g., stock prices) and need to adapt to change rapidly or autonomously
- Only need limited computing r esources
- Can be used to train systems on huge datasets that cannot fit in one machine's main memory=> Out-of-core learning chops the data into mini-batches and uses online learning techniques to learn from these mini-batches
- Capable of adapting rapidly to both changing data and autonomous systems
- Learning rate - how fast they should adapt to changing data: higher means learn rapidly, but also forget old data quickly / lower means less sensitvie to noise and outliers 
- The system's performance will gradually decline if bad data is fed to the system, need to monitor closely and promptly switch learning off and react to abnormal data
### #BatchLearning  = offline learning
- Cannot learn incrementally, model must be trained using all the available data, so offline
- Lot of time and computing resources
- Train a new version on the full dataset (old and new)
 