# Question 2
## Comparing time taken in both methods:
For m*m matrix with m=6000,\
Time taken to solve using np.linalg.solve = 5.040955305099487\
Time taken to solve using np.lianlg.inv = 31.0155029296875

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/117573435/fe45d866-649e-4605-bb54-c6ba2da1dc13)

***From the plot we can clearly see that as we increase the size of the matrix m, the time taken by the np.linalg.solve() is much less than the time taken by np.linalg.inv(). Although they give similar time when m is small because the computational time will be similar for small matrices.***
## Taking an Almost Singular Matrix and comparing both methods with Sk Learn's Result
Values of coefficients obtained by np.linalg.solve = [2.         1.21428571 0.39285714]\
Values of coefficients obtained by np.linalg.inv = [2.      1.1875  0.40625]\
Values of coefficients obtained by sklearn = [2.         1.24783188 0.37608406]

## Calculating  Mean Squared Error with respect to Sklearn's output
MSE using np.linalg.solve= 0.00046889391117500216\
MSE using np.linalg.inv= 0.0015166400078924598

## From the above results it is evident that np.linalg.solve is a more efficient method than np.linalg.inv:
*   np.linalg.solve has a much faster algorithm.
*   np.linalg.solve performs far better than np.linalg.inv on cases where the matrix is nearly singular.

## This is because :
*  np.linalg.solve is more stable as compared to np.linalg.inv when nearly singular matrix is fed, as it has special algorithms to handle that.
*  np.linalg.solve uses LU decomposition with partial pivoting as compared to computing the inverse in case of np.linalg.inverse.

**np.linalg.inv**


1.   The required matrix should be square and full rank matrix(Non singular matrix). So, if the matrix is singular means this will not be able to calculate the inverse of matrix.
2.   It computes the inverse of the matrix as follows: A is a square matrix, its inverse A^(-1) is such that A * A^(-1) = I, where I is the identity matrix.

3. The process of computing inverse of matrix using the method mentioned above is computationally expensive. As the computation of inverse calculation involves matrix multiplications and determinant calculation which will take much time if the matrix is a large one.
Thus it takes more time to compute the inverse of matrix as compared to np.linalg.solve() method.

**np.linalg.solve**

1.   In order to compute the inverse of a matrix, it solves the linear system of equations using methods like Gaussian elimination, LU decomposition, etc.These algorithms are optimized for numerical stability and computational efficiency, resulting in faster computation times compared to methods involving direct matrix inversion. Thus it takes less time to compute the inverse of a matrix using np.linalg.solve as compared to *np.linalg.inv*
2.   np.linalg.solve performs better when matrices are nearly singular. The reasons is following:

  np.linalg.solve calculates the inverse using the Gaussian elimination method or LU decomposition. So in case of Gaussian elimination or LU decomposition, the process of swapping row is done in such a way that the pivioting element has greater magnitude. This helps to prevent division by small numbers and reduces the amplification of numerical errors.

## Comparing on ill conditioned matrix
Ill-conditioned matrix A:\
[[1.00000000000000000000 1.00000000000000000000]\
 [1.00001000000000006551 1.00000000000000000000]]\

 B: [2, 3]\
Condition number: 400002.0000041729

Solution using np.linalg.solve: [99999.99999943548755254596 -99997.99999943548755254596]\
Solution using np.linalg.inv: [99999.99999943550210446119 -99997.99999943550210446119]

Changing B to [2 + 1e-3, 3]:\
Solution using np.linalg.solve: [99899.99999943606962915510 -99897.99899943606578744948]\
Solution using np.linalg.inv: [99899.99999943608418107033 -99897.99899943608033936471]

We can observe that with small change in B, the answers have been drastically changed. Also, the value of the results for np.linalg.solve and np.linalg.inv differ after around 10 decimal places. This is because of the difference in their method of computing the solution.
