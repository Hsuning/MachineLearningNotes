
How data is represented in Tensorflow
- Inconsistencies between how data is represented in numpy and in tensorflow
- 2 way of representing #Matrices : #NumPy  and #Tensorflow
- Tensorflow was designed to handle very large datasets
- Representing the data in #Matrices lets tensor-flow be a bit more computationally efficient internally
- When passing a numpy array into Tensorflow, it converts it to its own internal format tf.Tensor and operate efficiently using Tensor
- When read the data back, we can keep it as a tensor or convert it back to a numpy array


![](Pasted%20image%2020221212175304.png)
- tf.Tensor : a data type in Tensorflow to store and carry out computations on matrices efficiently, a way of representing #Matrices 
- a1.numpy(): take the same data and return it in the form of a numpy array