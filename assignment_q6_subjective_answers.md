# **Question 6**
***Compare DT, RF and Linear regression (yes, regression). For linear regression: each class label as an integer value. Say, 1: Sitting, 2:..., and so on. Use features extracted (from flattened out Linear Acceleration) using the TSFEL library. Compare the performance of these models. Is the usage of linear regression for classification justified? Why or why not?***
## Without Feature extraction
### Accuracy on Train Data {with shape (108 data points, 500 time series points)}:
Accuracy on training data for Decision Tree Classifier: 1.0\
Accuracy on training data for Random Forest Classifier: 1.0\
Accuracy on training data for Linear Regression Classifier: 1.0

### Accuracy on Test Data :
#### Decision Tree Classifier
Decision Tree Classifier Accuracy on Testing Set: 47.22%\
Decision Tree Classifier Precision on Testing Set: 43.55%\
Decision Tree Classifier F1_score on Testing Set: 41.77%\
Decision Tree Classifier Recall on Testing Set: 47.22%

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/e9a01c83-00f1-4bf0-97c4-9ec2a3308472)

#### Random Forest Classifier
Random Forest Classifier Accuracy on Testing Set: 52.78%\
Random Forest Classifier Precision on Testing Set: 47.58%\
Random Forest Classifier F1_score on Testing Set: 49.36%\
Random Forest Classifier Recall on Testing Set: 52.78%

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/4473bd19-afc2-494e-91db-134051f92e27)

#### Linear Regression based Classifier
Since we get continuous values from linear regression but our y_test is discerete integer values, we have rounded off the predictions.

Linear Regression Accuracy: 16.666666666666664 %

## After Feature Extraction
#### Decision Tree Classifier
Decision Tree Classifier Accuracy on Testing Set: 69.44%\
Decision Tree Classifier Precision on Testing Set: 70.16%\
Decision Tree Classifier F1_score on Testing Set: 69.50%\
Decision Tree Classifier Recall on Testing Set: 69.44%

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/cd4bb3dd-3116-488a-ab2d-e6a1c7f25c3f)

#### Random Forest Classifier
Random Forest Classifier Accuracy on Testing Set: 77.78%\
Random Forest Classifier Precision on Testing Set: 77.30%\
Random Forest Classifier F1_score on Testing Set: 77.20%\
Random Forest Classifier Recall on Testing Set: 77.78%

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/6ece8019-a1e2-4eee-aee8-75a1a8b1b1db)

#### Linear Regression Based Classifier
Linear Regression Accuracy: 13.88888888888889 %

# **Comparision of Accuracy**
*As per our results we can see that Random Forest Classifier is the best model.*

**Accuracy Comparision before feature extraction:**
*   *Accuracy for Decision Tree Classifier = 47.22%*
*   *Accuracy for Random Forest Classifier = 52.78%*
*   *Accuracy for Linear Regression Classifier = 16.66%*

**Accuracy Comparision after feature extraction:**
*   *Accuracy for Decision Tree Classifier = 69.44%*
*   *Accuracy for Random Forest Classifier = 77.78%*
*   *Accuracy for Linear Regression Classifier = 13.88%*

The results we are observing can be explained by understanding the nature of the algorithms used and the characteristics of your dataset.

### Accuracy of Decision Tree Classifier (69.44%) and Random Forest Classifier (77.78%):
Decision trees and random forests are both powerful classifiers commonly used for classification tasks. They work well when the relationships between features and the target variable are non-linear or complex.
The relatively higher accuracy of these classifiers suggests that the decision boundaries in our dataset are likely non-linear or complex, and these models are able to capture those relationships effectively.

### Accuracy of Linear Regression Classifier (13.88%):
Linear regression is primarily used for regression tasks, where the target variable is continuous. However, it can be repurposed for classification by thresholding the predicted continuous values.
The low accuracy of linear regression suggests that it's not well-suited for our classification task. Linear regression assumes a linear relationship between the features and the target variable, which might not be the case in our dataset.
Linear regression may not be able to capture the complexity of the relationships between features and the target variable, leading to poor performance in classification tasks, especially when the relationships are non-linear or the data is not well-suited for linear modeling.

## Is using linear regression justified here?

Based on the results and the characteristics of our dataset (where decision tree and random forest classifiers outperform linear regression significantly), it doesn't seem justified to use linear regression for this classification task.
Linear regression is better suited for regression tasks where the relationship between variables is linear, and it may not perform well in capturing non-linear relationships present in your dataset.
