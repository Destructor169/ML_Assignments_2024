# Question 5

## Image Reconstruction-
We are using Gradient Descent and Adam Optimizer for Reconstruction with rank r = 500.

###  Rectangular block of 30X30 is assumed missing from the image:

Input Image:

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/1c0b8787-88b3-40d0-a9ca-8ba90dd5a408)

Output Image (Gradient Descent):

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/22a42d10-ec81-404a-9561-e37e4bdd7f40)

Output Image (Adam Optimizer):

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/9cac1caf-ab4c-4507-a554-3c887e93e5e1)

Comparing patches (From Left to Right -> Original Patch, Gradient Descent, Adam Optimizer):

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/6760c45e-790c-4bd8-aecd-eadd48efa928)

Loss metricts observed for above:\
Gradient Descent:\
PSNR = 30.185585220994355\
RMSE = 0.030954282296610954

Adam Optimizer:\
PSNR = 38.79101168267138\
RMSE = 0.011493423668080604

Observation:\
The results suggest that while Matrix Factorisation can reconstruct images with missing patches, the quality of the reconstruction may be moderate, and there may be noticeable differences between the original and reconstructed images. We can observe that the patches are not reconstructed properly.
### Random subset of 900 (30X30) pixels is missing from the image:
Input Image:

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/55cf79c1-7884-4b28-946e-0072e66fcb4f)

Output Image (Gradient Descent):

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/207a7ee2-cc21-4477-a012-85d00cc4aa2f)

Output Image (Adam Optimizer):

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/6b348714-9472-483f-8fd5-a02612645c31)

Loss metrics observed for above:\
Gradient Descent:\
PSNR = 30.3583450584107\
RMSE = 0.030344692932033075

Adam Optimizer:\
PSNR = 42.20405450323096\
RMSE = 0.007758848552920304

Observation:\
The results here are much better than in missing patch version. We cannot easily detect differences between the original and reconstructed image. This is because the pixel values that are missing can be predicted based on the surrounding pixels. In case of missing patch, we don't have access to pixels that are close to the missing pixels. While in randomly missing, almost all missing pixels are between known pixels. Because of this, the reconstruction is better in Randomly missing data than in Rectangular patch.

### Using RFF + Linear Regression:
We are using the same image for reconstruction using RFF and linear regression. We have used 15000 RFF features, trained over 1000 epochs.
#### Rectangular missing patch
Output Observed:

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/117573435/19eba191-6069-4d9f-8901-4cdf5157e187)

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/117573435/cca4557f-65a1-4e62-956a-27d84328e3f9)

Loss metrics observed for RFF implementation:\
PSNR = 23.325772978327425\
RMSE = 0.06818853361208614

#### Random Subset Removed
Output Observed:

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/120514790/0ac25231-f2c2-4546-8026-85064e38ba35)

Loss metrics observed for RFF implementation:\
PSNR = 29.846642725405545 \
RMSE = 0.03218606349705193

For the given rank and number of features, we observe that Matrix factorisation performs better than RFF + Linear Regression in both cases. Linear Regression with RFF takes a significantly higher amount of computation as compared to matrix factorisation for the chosen values of rank and features. Therefore, computationally, matrix factorisation is a better option for image reconstruction.
## Varying Region size :
Output plots:\
(From Left to Right -> Input Rect. patch, Reconstructed Rect. patch, Input Random subset, Reconstructed Random subset)\
(From Top to Bottom -> Patch Sizes = {20,40,60,80} )

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/3b068edf-58d2-463b-8c10-31ca331217bc)

Loss metrics observed with varying region size:

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/ea299928-28e9-45eb-af8c-1bba90cd6237)

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/b1a35c0d-fced-4843-9d78-eeb14dcf9829)

Observations:\
With increasing region size, in both the cases, we can see an increase in RMSE and decrease in PSNR indicating poorer performance of the model. This is understandable as with increasing missing region size, the available data is reduced and thus the reconstruction gets worse. This change observed is very less in case of Randomly removed pixels whereas the change is drastic for missing patches. This is because with increasing patch size, the missing data is more and more uncorrelated to the available data whereas in random subsets, even if more pixels are missing, they are related to the surrounding pixels which are available to us. 

## Using Alternating Least Squares:
Alternating least squares method takes much longer to train than Gradient Descent or Adam Optimizer. Therefore we have used rank as 75 instead of 500. 

### Rectangular Patch
Input Image:

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/308be8fe-c6f1-4860-bdc3-71eda8bfee5c)

Reconstructed Image:

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/27f2173a-a37e-4190-b486-8c6298bc4741)

Comparing Patches (Original on left):

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/d982548d-da81-4731-95e3-a79bcfbb180e)

PSNR = 23.539679061712704\
RMSE = 0.06652977380903954

We observe that the reconstructed patch contains almost random colors. But the remaining image is correctly formed.
### Random Subset Removed
Input Image:

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/6a7cb161-3d63-4868-8fd3-884948f21f4f)

Reconstructed Image:

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/83ad4806-4984-44a0-9ee0-50dcfbf24e4b)

PSNR = 23.916079753919803\
RMSE = 0.06370829938886696

We observe that in case of Randomly removed data, the reconstructed image seems correctly formed.
### Varying Region Size:
Output plots:\
(From Left to Right -> Input Rect. patch, Reconstructed Rect. patch, Input Random subset, Reconstructed Random subset)\
(From Top to Bottom -> Patch Sizes = {20,40,60,80} )

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/58619508-a334-48b5-8f86-a3083d824f3f)

Metrics observed vs Region size in Alternating Least Squares method:

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/9da003fd-8637-430b-8ae4-817062350398)

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/146793425/1499b05b-aa9e-437d-ac69-015797d79f93)

In comparison to Gradient descent, Alternating Least squares is performing poorly for missing patches.

## Data Compression-
Original Image :

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/117573435/49854fc4-1bdb-416b-8bab-e8371a6253d3)

Images after removing patches for the three cases i.e:\
a patch with mainly a single color.

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/117573435/70702ec4-476e-4219-ae00-7e26487ca989)

a patch with 2-3 different colors.

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/117573435/1b140b84-32fd-4b79-952b-f1ef897368ef)

a patch with at least 5 different colors.

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/117573435/c6a011d6-97ad-4d31-81b5-9453f58f6059)

Predicted patches and original patches are shown below:

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/117573435/3b2406d5-c6c7-4c1c-8ec7-38e34cf8beab)

Plotting the reconstructed patches over the original image (retaining all pixel values outside the patch, and using your learnt compressed matrix in place of the patch)

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/117573435/abb391db-46c0-47ff-978d-8627f685a473)

RMSE for the predicted patches is shown in the plot below:\
For single-colored patch:

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/117573435/27a62135-3604-4d4b-a523-7f78a478904b)

For patch with 2-3 different colors:

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/117573435/c8294ed3-9634-4857-8922-9aa7edc68d9f)

For patch with at least 5 colors:

![image](https://github.com/ES335-2024/assignment-2-es-335-2024-ml-mavericks/assets/117573435/3f4dca5c-2889-4021-b6f5-cb2754afbda4)

Inferences:\
We can see that RMSE for single-colored patch is very less as compared to that of multi-colored because it is easy to predict single-colored patch comparatively.
Also, with an increase in rank, prediction becomes better due to which RMSE decrease for all patches. But for same rank, (RMSE of single-colored patch) < (RMSE of 
patch with 2-3 different colors) < (RMSE of patch with at least 5 colors). The same can be observed qualitatively from the plot for predicted patches. Also, this problem shows the application of image compression and we can see that at lower ranks prediction is a good approximation of ground truth. At higher ranks, predicted and original patches cannot be distinguished by the naked eye.
