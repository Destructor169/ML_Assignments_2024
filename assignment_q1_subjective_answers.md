# Question 1
1. Generate the following two functions:

    Dataset 1:
    ```python
    num_samples = 40
    np.random.seed(45)
        
    # Generate data
    x1 = np.random.uniform(-20, 20, num_samples)
    f_x = 100*x1 + 1
    eps = np.random.randn(num_samples)
    y = f_x + eps
    ```
    
    Dataset 2:
    ```python
    np.random.seed(45)
    num_samples = 40
        
    # Generate data
    x1 = np.random.uniform(-1, 1, num_samples)
    f_x = 3*x1 + 4
    eps = np.random.randn(num_samples)
    y = f_x + eps
    ```

- Implement full-batch and stochastic gradient descent. Find the average number of steps it takes to converge to an $\epsilon$-neighborhood of the minimizer for both datasets. Visualize the convergence process for 15 epochs. Choose $\epsilon = 0.001$ for convergence criteria. Which dataset and optimizer takes a larger number of epochs to converge, and why? Show the contour plots for different epochs (or show an animation/GIF) for visualisation of optimisation process. Also, make a plot for Loss v/s epochs. **[2 marks]**
- Explore the article [here](https://machinelearningmastery.com/gradient-descent-with-momentum-from-scratch/#:~:text=Momentum%20is%20an%20extension%20to,spots%20of%20the%20search%20space.) on gradient descent with momentum. Implement gradient descent with momentum for the above two datasets. Visualize the convergence process for 15 steps. Compare the average number of steps taken with gradient descent (both variants -- full batch and stochastic) with momentum to that of vanilla gradient descent to converge to an $\epsilon$-neighborhood of the minimizer for both datasets. Choose $\epsilon = 0.001$. Write down your observations. Show the contour plots for different epochs for momentum implementation. Specifically, show all the vectors: gradient, current value of theta, momentum, etc. **[2 marks]**

# First Half

## 1- Comparing average steps to converge:
Results are as follows-\
Average Steps to Converge (Dataset 1): {Learning Rate = 0.0001}\
Full Batch: 648.0 iterations, 648.0 epochs\
Stochastic: 25670.94 iterations, 641.16 epochs

Average Steps to Converge (Dataset 2): {Learning Rate = 0.02}\
Full Batch: 10.0 iterations, 10.0 epochs\
Stochastic: 19588.18 iterations, 489.26 epochs

Average Steps to Converge (Dataset 1): {Learning Rate = 0.001}\
Full Batch: 2500.0 iterations, 2500.0 epochs\
Stochastic: 2367.24 iterations, 58.61 epochs

Average Steps to Converge (Dataset 2): {Learning Rate = 0.001}\
Full Batch: 186.0 iterations, 186.0 epochs\
Stochastic: 7397.28 iterations, 184.41 epochs

### Observations and Inference:
- Dataset 1
  In dataset 1, with small learning rate, both Full batch and Stochastic Gradient Descent converge around 640 epochs, SGD converging earlier than Full Batch.\
  Though in terms of epochs SGD is performing slightly better, if we compare in terms of iterations, then surely Full Batch performs better.\
  On higher learning rate, in this case at 0.001, Full Batch gradient descent has diverged (Limit on no. of epochs is 2500). On the other hand, performance of SGD has greatly improved, from 641 epochs to 58 epochs.\
  This is because Higher learning rate causes SGD to take greater steps at each iteration and as it calculates gradient at each datapoint, higher learning rate helps it to escape the noisy gradients causing it to converge faster.\
  Whereas, if Full batch, the higher learning rate causes it to diverge because it moves in the overall direction of the true gradient, therefore because of high learning rate, it may overshoot or skip the minima and diverge.\
  Thus in this case, a higher learning rate is advisable for SGD but not for Full Batch Gradient descent.

- Dataset 2
  In dataset 2, with small learning rate (0.001), both full batch and stochastic gradient descent converge around 185 epochs, SGD again converging a little earlier than Full Batch.\
  But on higher learning rate (0.02), Full Batch Gradient descent has converged very quickly, while SGD took a lot longer to converge than before i.e. 490 epochs. We observe a change in behaviour of the two optimizers with respect to the learning rates in dataset 2. This difference in the behaviour of the two datasets with respect to the learning rates can be explained as follows:
  - In both the algorithms, we are initializing the parameters $\theta$ with zeros.
  - The two datasets have optimal values for $\theta$ as (100,1) and (3,4) respectively.
  - Therefore, for dataset 1, the initial $\theta$ is much farther from optima than for dataset 2, because of which the gradient values for dataset 1 are high compared to dataset 2. Hence, the learning rate required for dataset 1 is smaller compared to dataset 2.
  - The requirement of different learning rates for 2 dataset is causing the difference in the behaviour of the algorithms. 

## 2- Plotting the convergence process
After experimenting a little, we found good learning rates as follows: 
learning_rate for dataset 1 (Full Batch) = 0.001\
learning_rate for dataset 1 (Stochastic)= 0.0001\
learning_rate for dataset 2 (Full Batch) = 0.1\
learning_rate for dataset 2 (Stochastic) = 0.02\
For proper visualization, we will be using the above learning rates in the contour plots.\
Contour plots:

![image_contour_plot_dataset1](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/0e05cacb-38ab-4feb-acc7-fd3758ffbf00)

![image_contour_plot_dataset2](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/eb196a04-d449-433e-aeef-2d519d9bc725)

### Observations and Inference:
Observing the convergence procedure for 15 epochs shows us that in gradient descent, we are smoothly moving directly towards the point of minima. While in SGD, we seem to be going in various directions but overall we progress towards the minima only. This is visible as noisy path traced by the SGD optimizer. Though SGD's plot is noisy, it is overall moving towards the point of minima in same manner as Full Batch GD.\
This demonstrates how SGD is unbiased estimate of the true gradient because on average, they move in the same directions.

## 3- Plotting Loss vs Epochs
Following is the plot of Loss vs Epochs:

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/0d2f406c-f893-47cd-ad7f-094e00f41e33)

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/d776975d-9751-463c-95c6-f2e53f444ef9)

In the above 2 plots, the curve for SGD appears smooth because we are only seening points after each epoch and not the intermediate losses after each iteration. 
Also, we can observe that initially, the convergence rate of SGD is very fast compared to Full Batch GD. This is because initially when loss is high, all the data points have gradients pointing towards the minima and we make N number of updates in each epoch because of which we converge faster than full batch, where we only make one update per epoch.

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/930b7d69-074e-403e-a459-36fe3712fea8)

Here we can see the noisy curve for SGD when we plot the loss vs iterations. This is because the gradients with each data point can point in various directions and hence many times we make updates in wrong directions which we correct later. Because of this, the curve turns out to be noisy.

# Second Half (With Momentum)
## 1- Computing average steps to converge
Results:\
Average Steps to Converge (Dataset 1): {Learning Rate = 0.0001}\
Full Batch with momentum: 2578.0 iterations, 2578.0 epochs\
Stochastic with momentum: 2678.65 iterations, 66.35 epochs\

Average Steps to Converge (Dataset 2): {Learning Rate = 0.02}\
Full Batch with momentum: 35.0 iterations, 35.0 epochs\
Stochastic with momentum: 48.85 iterations, 0.61 epochs

### Observations and Inference
From the above results, we observe that:
- Adding Momentum has caused the Full Batch Gradient Descent to take more epochs than previously in both datasets (from 648 and 10 epochs to 2578 and 35 epochs respectively).
- On the other hand, adding momentum to SGD has greatly improved its performance on both datasets (from 641 and 489 epochs to 66 and 0.61 epochs). In the case of Database 2, it often converged even before seeing the complete data i.e before 1 epoch.
The above results can be explained in following manner:
- Addition of momentum in Full Batch Gradient Descent is causing the optimizer to overshoot the minima. As we get near the minima, we gain some momentum and at minima, because of the momentum, we overshoot and then slowly return back because of the gradient. This is similar to a ball oscillating when dropped from a height in a bowl. Because of this overshooting, the number of epochs required to converge has increased significantly.
- Whereas, addition of momentum in SGD has opposite effect. As in SGD, the gradients at each iteration are noisy and not similar. Therefore, the momentum does not continuously add up to make it overshoot. Instead, the momentum smoothens the overall convergence procedure. The small noises at each gradient get cancelled out because of the added momentum term. This helps the SGD to continuously approach the minima without overshooting or deviating from the correct path due to noisy gradients. Hence adding momentum in SGD has increased its performance.

## 2- Contour Plots
Following are the contour plots for 15 epochs:

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/b8bd8109-3a71-4b22-8fd4-1a4e52c81aff)

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/b8a49ec9-44d4-459e-a4d1-16b78ac155fa)

We can observe that Full Batch GD is overshooting as it oscillates around the minima while SGD has a smoother curve than before and is converging easily.
### Contour Plots with Change vectors:
Dataset 1:\
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/09ed18dd-148a-4538-bfae-055015b05731)
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/7b56f01a-d15f-493d-be99-f4c3f2427ac8)
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/aa10cb26-b0c7-4c55-84de-2191e80536a4)
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/886f6b81-e546-47c0-8f50-55e4193cc501)
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/08c135c7-c4f4-4554-b981-b2f2d8678228)
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/098c13d9-8dfa-4191-885d-98f2a554f17a)
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/4893bb6d-c78c-4a29-bb1c-01205e91c1ff)
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/6411d8c8-4c7e-4c7a-90d2-be048d7bb7f7)
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/4558e1af-555c-4818-a911-0d12dca5129e)
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/1e94f538-e89e-4b7e-8aef-97da497cd673)
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/369826af-7e6c-4244-8c40-2a44142e05c0)
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/ce8ed5c3-ad19-426f-898e-0d2893af4986)
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/91e0ed86-d6bf-4d4e-9a91-e89d4996f12b)
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/759cacf8-0029-4c20-80e0-e2d8e252d7ed)
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/faf03483-343c-41a6-92c2-07dcc2b19920)

Dataset 2:\
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/1aa6f664-47cf-4213-a23a-78a182bd6d1c)
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/a3098942-8059-4b97-a6c3-46d164391f38)
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/2d8ae0b4-ecb6-4951-8f03-d3230627ca02)
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/da372459-ca2f-43ab-a5ee-e5d1eb2282f2)
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/6d7bf081-01e6-492b-a765-9946d0e5424f)
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/9315c479-ebed-4a73-9d9b-0489320fc635)
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/d0b10866-d152-4238-9306-a4404a19653d)
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/5640f270-0c09-474e-90a7-6cc8e7d1ea95)
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/34009bab-2471-4510-8033-7622b4e5c51f)
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/2883314d-a5a2-4b2e-8ab7-b5c83f8bac03)
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/488b9a76-ce7f-411e-acb2-76b49281faa8)
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/47daf075-ff1d-4239-ad50-02710726ad78)
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/695591fb-4ac1-40f4-aadf-2c430e43e6ed)
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/510c54c7-09b5-43c7-ad1c-a832d1f20929)
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/dac3dac0-a7c4-44b0-b01c-d11c0cc1889b)

We can observe from the above plots how in Full Batch Gradient Descent, the change vectors sizes are increasing and then after crossing the minima, they start decreasing. Whereas in SGD the changes remain small and there is less deviation because of the momentum.

Following are plots of the gradients and momentum vectors:

Full Batch:\
Dataset 1\
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/56bdd1ce-3f46-4faf-acaf-6be50bd207b2)

Dataset 2\
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/9cce8eb1-563d-4859-8ace-9d6efa140bc9)

Stochastic:\
Dataset 1\
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/c14b4979-2816-44a0-a726-4a77f976bb2d)

Dataset 2\
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/1ecbb74f-01c9-4e2f-8179-75269e874ad6)

The plots show the same inferences as before. For Full Batch GD, the momentum keeps getting added because all are in same direction causing it to overshoot, whereas for SGD, the gradient vectors point in different directions and hence the added momentum make it to continue moving without deviating for noisy gradients.

## Conclusion
In this Question, we have seen the various differences between the Stochastic gradient descent and Full batch gradient descent methods. We contrasted among them on the basis of their performance for two types of dataset requiring different learning rates. We also saw the effects of momentum on the two optimization techniques. 
