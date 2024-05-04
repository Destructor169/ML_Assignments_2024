# Question 4A: Superresolution (Qualitative comparison)

The cropped image that we have used for superresolution has dimensions 200 X 200. Number of RFF features used = 8000.

## Original image and Predicted image(200 X 200):

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/117573435/afa7d9b2-8da3-48c6-8801-a6b708f3958a)

## Original image and Superresolved Predicted image(400 X 400):

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/117573435/5ddbb0e0-07f2-4abc-9f03-10c8b080ec3a)

## Predicted image(200 X 200) and Superresolved Predicted image(400 X 400):

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/117573435/d4aa7dbd-6055-4df5-8ef3-6c1af0301767)

We can observe that reconstruction is not that good as we have used only 8000 RFF features due to limited hardware resources. Qualitatively, we can see each pixel of the original image by zooming in on the image as it is a 200X200 image but it is not possible in the case
of the super-resolved predicted image as it is a 400X400 image. Also, we cannot observe much difference in the two predicted images even though they have different resolutions.

# Question 4B:  Superresolution (Quantitative comparison)

Number of RFF features used = 8000

# Ground Truth High Resolution Image(400X400):

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/117573435/5d2cc55b-f63b-416b-99b8-83abc9f2b7dd)

# Input Image(200X200):

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/117573435/ecc385f3-0167-4a08-b824-29681d8ddfe2)

# Predicted High-Resolution Image(400X400):

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/117573435/a4a40902-361b-4bba-a188-6e1c222beb5f)

RMSE of predicted and ground truth high-resolution image = 0.122

Peak SNR of predicted and ground truth high-resolution image = 18.255

As in the previous part, since the number of RFF features used is 8000, the prediction is not good. It is also evident quantitatively as Peak SNR is only 18.255 and RMSE is 0.122. 

# Question 4C: Complete image with random missing Data
The cropped image that we have used for superresolution has dimensions 200 X 200. Number of RFF features used = 10000.
## Original image(200 x 200)
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/120514790/c4340e7d-4518-4a1b-a5c0-1883ccbfb100)

We utilized Random Fourier Features (RFF) with 10,000 features for image reconstruction. Unfortunately, due to hardware constraints, we were unable to work with a higher number of features. However, we anticipate that increasing the number of features would lead to improved results. Our experimentation with 8,000 and 10,000 RFF features showed enhanced prediction accuracy and reduced blurriness in the results.

The following are the results obtained:

## 10% Data removed
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/120514790/806d6321-a50d-4a6a-b7ce-15a2c04b337b)

## 20% Data removed
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/120514790/a3f1fd0c-9b39-4ecf-bdbf-a32e594af033)

## 30% Data removed
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/120514790/b244e77d-1b16-43bf-b72e-2180ac4ef0dc)

## 40% Data removed
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/120514790/9bc07ae4-2090-4437-b9e6-8b8f6d05b9ca)

## 50% Data removed
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/120514790/58cef4f4-9f65-4482-a2a1-ca49a3e8717a)

## 60% Data removed
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/120514790/d7040122-ee47-4d87-b4d4-6670963a695a)

## 70% Data removed
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/120514790/e0b860fc-e05e-4a2b-9bef-72ac8862b71a)

## 80% Data removed
![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/120514790/53fcbc69-032f-4da3-b0e2-44f628b46058)

## 90% Data removed

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/120514790/956c29a7-2663-4eeb-aa18-aa4d75436ea0)

When more than 50% of the data is removed from the original image, the image becomes unrecognizable to the human eye. However, through our reconstruction process, we were able to achieve decent results where the dog in the image is recognizable. This outcome is truly remarkable.

# Metrices
We have calculated the Root Mean Square Error (RMSE) between the predicted and ground truth high-resolution images, as well as the Peak Signal-to-Noise Ratio (PSNR) values. Our findings indicate that as more data is removed, the RMSE value increases while the PSNR value decreases. This trend is visually represented in the accompanying graph.

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/120514790/a2521493-84b9-4d31-ba50-a15832f5c626)

The RMSE values remain relatively stable and low from 0.1 to about 0.5 data removed, suggesting that the model's performance is consistent despite the removal of up to 50% of the data.
However, there is a noticeable increase in RMSE from 0.6 to 0.9 data removed, with a sharp rise between 0.8 and 0.9. This indicates that the model's prediction error increases significantly when more than 60% of the data is removed, with the greatest loss in accuracy occurring after 80% of the data is removed.

The PSNR values start high when only 10% of the data is removed and gradually decrease as more data is removed.
The decrease in PSNR is relatively slow and linear from 0.1 to about 0.5 fraction of data removed, but it becomes more pronounced from 0.6 onwards, with a steep decline between 0.8 and 0.9 data removed. This suggests that the quality of the reconstruction degrades significantly when more than 60% of the data is removed, with the most substantial degradation occurring after 80% of the data is removed.

### Inferences:

Both RMSE and PSNR metrics indicate that the system or model being evaluated is robust to a certain level of data removal (up to 50%).
The quality of the model's output or the reconstruction quality deteriorates significantly when more than 60% of the data is removed, with the most critical impact observed after 80% data removal.
The sharp increase in RMSE and the sharp decrease in PSNR after 80% data removal suggest a threshold beyond which the model or system fails to maintain its performance and accuracy.
Decisions regarding data removal should consider the impact on these metrics, especially if maintaining low RMSE and high PSNR is critical for the application in question.
