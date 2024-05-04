# **Question 7**
*Obtain the weights (take absolute values as weights can also be negative) of the linear regression model. Also, obtain the feature importance from the Random Forest model. Plot the weights obtained as a Bar plot. This will help you visualize what features are being prioritized by the models. Note that sum of feature importances for a Random Forest model is 1. you will have to bring the linear regression weights to the same scale. To do so you can divide the weights by the sum of all the weights. Plot the importance of the features in the same plot. Figure out the top 10 important features obtained from both the models and display their names. What do you infer?*

## Bar Plot for weights obtained in Linear Regression {Without Normalisation}:
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/dabd38a3-210c-4ba6-8870-86d79f3319e5)

## Bar Plot for feature importance from the Random Forest model:
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/c8c36d12-98ff-46a5-9633-b15dbd7a7340)

## Bar Plots of weights {After Normalisation} and importances:
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/fab4defd-87e6-4108-8af2-75bc802664ef)

## Plotting on same Plot:
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/f138d0a2-ea50-44ff-bba9-4e60e8e18dc1)

## Top 10 Features:
### Using Importance from Random Forest
0_Wavelet energy_8\
Importance = 0.029298854464300365 

0_Median\
Importance = 0.023286635375842097

0_Wavelet standard deviation_7\ 
Importance = 0.0209292213917098 

0_Absolute energy\
Importance = 0.019295073229528326 

0_ECDF Percentile_0\
Importance = 0.017147825364024948 

0_Wavelet energy_4\
Importance = 0.017141130506808856 

0_Autocorrelation\
Importance = 0.01704277210843387 

0_Wavelet standard deviation_8\
Importance = 0.016623198791963424 

0_Average power\
Importance = 0.016170824777995144

0_Wavelet standard deviation_3\
Importance = 0.01523212673565471

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/75d82d45-22fb-401d-9f16-b390e82ee4a6)

### Using Weights from Linear Regression
0_Mean absolute deviation\
Weight = 0.032017971637332276 

0_FFT mean coefficient_9\
Weight = 0.025283698702276043 

0_FFT mean coefficient_79\
Weight = 0.020397365931956325 

0_FFT mean coefficient_71\
Weight = 0.02016344918640772

0_FFT mean coefficient_46\
Weight = 0.019757326873023277

0_FFT mean coefficient_53\
Weight = 0.01801717850481024 

0_FFT mean coefficient_8\
Weight = 0.017925723893177783

0_Variance\
Weight = 0.017771099551326566 

0_FFT mean coefficient_75\
Weight = 0.015798272798176922

0_FFT mean coefficient_88\
Weight = 0.015758214721985256

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/6571f9ac-dd50-48d6-8638-953151093fd0)

## Conclusion:
We have observed that the top important features for Linear Regression are completely different from the top important features for Random Forest.
### Linear Regression prioritizes features with higher weights:
Linear regression assigns weights to each feature, indicating their importance in predicting the target variable. Features with higher absolute weights are considered more important by the linear regression model.\
Features that have a strong linear relationship with the target variable are likely to have higher weights in linear regression.
### Random Forest prioritizes features based on information gain:
Random Forest calculates feature importances based on how much each feature contributes to decreasing impurity (e.g., Gini impurity or entropy) in the decision trees within the forest.\
Features that are effective in splitting the data into homogenous classes at decision tree nodes are considered more important by the random forest model.
### Different models capture different types of relationships:
Linear regression assumes a linear relationship between features and the target variable, so it tends to prioritize features with linear dependencies.
Random Forest, on the other hand, can capture non-linear relationships and interactions between features more effectively due to its ensemble nature and ability to fit complex decision boundaries.
