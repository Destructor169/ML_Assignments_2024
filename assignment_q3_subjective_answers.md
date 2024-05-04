### Least-squares solution solves the equation Ax = b as closely as possible, in the sense that the sum of the squares of the difference b − Ax is minimized.

### In scikit-learn, the least-square solution is used by the singular value decomposition (SVD) method to compute the inverse efficiently. If the matrix is full rank, the least-square solution is same as the normal solution (calculation based on general matrix operations).

### This type of inverse is called the Moore-Penrose inverse, also known as the pseudoinverse, is a generalization of the matrix inverse for non-square matrices or matrices that are singular or not full rank.

### Given a matrix $X$,  the Moore-Penrose inverse denoted as $X^+$ where $X^+$ $X$ $X^+$ = $X^+$ and $X$ $X^+$ $X$ = $X$

### SVD decomposes the design matrix $X$ as: $X$ = $U$ $D$ $V^T$ for some isometries $U$ and $V$, where  $V^T$ is transpose of $V$ and $D$ is a diagonal matrix containing the singular values of $X$. The pseudoinverse is then computed as: $X^+$ = $V$ $D^+$ $U^T$, where $U^T$ is transpose of $U$ and $D^+$ is the pseudoinverse of $D$, obtained by taking the reciprocal of each non-zero element on the diagonal of $D$ and then transposing the resulting matrix.

### This method allows scikit-learn to handle cases where the design matrix $X$ is not invertible due to multicollinearity or other issues, ensuring robustness in linear regression fitting even in challenging scenarios.





```python
import numpy as np
from sklearn.linear_model import LinearRegression

```


```python
x_1=[1,1,1]
x_2=[1,2,3]
x_3=[2,4,6]
X=np.array([x_1,x_2,x_3])# Feature Matrix
y=[4,6,8]# Target Array

```

### Checking the Rank of X


```python
r_x=np.linalg.matrix_rank(X)
print(f"Rank of Matrix X = {r_x}")
print(f"Since Rank of X = {r_x}, it is not full-rank (3)")
```

    Rank of Matrix X = 2
    Since Rank of X = 2, it is not full-rank (3)



```python
lr=LinearRegression(fit_intercept=False)#  Fit intercept=False because we have already included a columns of 1 in X for intercept.
lr.fit(X,y)
theta_sk=lr.coef_# Values of the coefficients obatined using Sk Learn
print(theta_sk)
```

    [ 3.13333333  1.33333333 -0.46666667]


### The normal equation provides the least-square solution for the linear regression problem.

### Solving using Normal Equation.


```python
def solve_normal_equation(X, y):# Solve using normal equation
    try:
        theta = np.linalg.inv(X.T @ X) @ X.T @ y
        return theta
    except np.linalg.LinAlgError:
        print('The matrix is singular.')
        print("X.T @ X = \n", X.T @ X)
        return None
solve_normal_equation(X,y)

A=(X.T)@X# Non-Invertible
```

    The matrix is singular.
    X.T @ X = 
     [[ 6 11 16]
     [11 21 31]
     [16 31 46]]


### The main problem is in computing the inverse of $X^T$ $X$. Let us compute the pseudo inverse of $X^T$ $X$ by using np.linalg.pinv() function.


```python

A_inv=np.linalg.pinv(A)# Computing the psuedo inverse of the Non-Singular Matrix A
theta_ps=A_inv@(X.T)@y# Used Psuedo Inverse of A as the inverse of X'X in the normal equation
print(theta_ps)
```

    [ 3.13333333  1.33333333 -0.46666667]


### Hence, libraries like SKLearn use Psuedo Inverse to compute the inverse of the non-invertible matrix.
