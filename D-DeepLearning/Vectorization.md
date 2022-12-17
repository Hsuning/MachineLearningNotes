# How Neural Networks Are Implemented efficiently
#Vectorization
```toc
```

## How Vectorized Implementation of Neural Networks Works
Very important for deep learning to success and scale today

![](Pasted%20image%2020221213123656.png)
- np.matmul: matrix multiplication
- for loop can be replaced with just a couple lines of code

## Matrix Multiplication (np.matmul)
- meaning of multiply two matrices

### Dot products
#DotProduct  
![](Pasted%20image%2020221213124603.png)
- $\vec{a}^{T}$: turn the vector into a row from column vector to row vector, by taking the transpose of a
- $z = \vec{a} \cdot \vec{w}$ (dot product) is the same as $z = \vec{a}^T\vec{w}$ (multiply)

### Transpose
#Transpose
- take the vector $\begin{bmatrix} 200 \\ 17 \end{bmatrix}$ and lay is elements on the side like this $\begin{bmatrix} 200 & 17 \end{bmatrix}$
- for a matrix, take the columns and lay the columns on the side, one column at a time:
	- $\begin{bmatrix} 1 & -1 \\ 2 & -2 \end{bmatrix}$ => $\begin{bmatrix} 1 & 2 \\ -1 & -2 \end{bmatrix}$

### Vector Matrix multiplication
#VectorMatrixMultiplication  
![](Pasted%20image%2020221213124733.png)

### Matrix Matrix multiplication
#MatrixMatrixMultiplication  
![](Pasted%20image%2020221213125545.png)
- when seeing a matrix, think of the columns of the matrix
- when seeing a transpose, think of the rows of that matrix
- $\vec{a}_1^T$ : first **row** of a transpose
- $\vec{w}_1$: first **column** of matrix
- take the first row of

### Matrix Multiplication Rules
![](Pasted%20image%2020221213140519.png)
 - $A^T$ : 3 x **2**
 - $W$: **2** x 4
 - we can only take dot products between vectors that are the same length
 - the number of **columns** of the first matrix transposed is equal to the number of **rows** of the second matrix

### The Output of Matmul
- same # rows as $A^T$
- same # columns as $W$

## Matrix Multiplication in Numpy
![](Pasted%20image%2020221213141316.png)
- A.T : transpose function in Numpy
- np.matmul is clear than using @, it can be done very efficiently using fast hardware
