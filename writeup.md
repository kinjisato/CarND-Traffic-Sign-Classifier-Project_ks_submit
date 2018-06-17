# **Traffic Sign Recognition** 

## Writeup

### Kinji Sato 17/June/2018

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./ImagesForWriteup/SummaryExploration001.jpg "Summary and Exploration"
[image2]: ./ImagesForWriteup/DistributionTrainingSet.jpg "Distribution Training Set"
[image3]: ./ImagesForWriteup/DistributionValidationSet.jpg "Distribution Validation Set"
[image4]: ./ImagesForWriteup/DistributionTestSet.jpg "Distribution Test Set"
[image5]: ./test_images/9_NoPassing.jpg "Traffic Sign 1"
[image6]: ./test_images/12_PriorityRoad.jpg "Traffic Sign 2"
[image7]: ./test_images/13_Yield.jpg "Traffic Sign 3"
[image8]: ./test_images/14_Stop.jpg "Traffic Sign 4"
[image9]: ./test_images/28_ChildrenCrossing.jpg "Traffic Sign 5"
[image10]: ./test_images/31_WildAnimalsCross.jpg "Traffic Sign 6"
[image11]: ./test_images/40_RoundaboutMandatory.jpg "Traffic Sign 7"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

Here is a link to my [project code](https://github.com/kinjisato/CarND-Traffic-Sign-Classifier-Project_ks_submit/blob/master/Traffic_Sign_Classifier-ks006.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used basic python code, and those were enough for questions.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. As first, I picked one picture up from each classes, and also wrote the meaning of the signs from signnames.csv.

![alt text][image1]

And second, I made distribution graphs for Training set, Validation set and Test test to confirm how many signs were included for each classed in the 3 sets.

![alt text][image2] ![alt text][image3] ![alt text][image4]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first try, I decided to convert the images to grayscale as same as the lectures, image size was converted to 32 x 32 x 1 from original 32 x 32 x 3. And convolution was also same as lecture. But this did not give good validation accuraty, that was below than 90%.
So, second try, I used nomalize filter for 3 color images. I applied normalize filter for each color channels, and after that I applied conv filter. The conv filter input size was increased to fit 3 colors from 1 grayscale. This gave better result than my first try of grayscale, and then I decided to tune conv filter as described below.

Without this normalization for 3 color channels, I did not use any other pre-processes for given images.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image (normalized)  							| 
| Convolution 3x3     	| 1x1 stride, padding 'VALID', outputs 28x28x6 	|
| RELU					|	Drop Out (0.9 for training)											|
| Max pooling	      	| 2x2 stride, padding 'SAME'  outputs 14x14x6 				|
| Convolution 3x3	    | 1x1 stride, padding 'VALID', outputs 10x10x16 	|
| RELU					|	Drop Out (0.9 for training)											|
| Max pooling	      	| 2x2 stride, padding 'SAME'  outputs 5x5x16 				|
| Flatten		| output = 400       									|
| Fully connected		| output = 120       									|
| Fully connected		| output = 84       									|
| RELU					|	Drop Out (0.9 for training)											|
| Fully connected		| output = 43 (same as classes of sign )      									|
| Softmax				|         									|

 The model structure is the almost same as the structure of lenet used in the lecture.
 Difference was drop out (keep_prob) to prevent overfitting when training.


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Optimizer : Adam
Batch size : 128
Epochs : 15
Learning rate : 0.001

Those are the almost the same as the lenet lecture. I choose 15 for epochs to check the validation accuracy becomes good. Other parameters, I don't have time to try (due to my laptop peformance). But, I think Adam optimizer, 128 for batch, 0.001 for lr would be ok.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.996
* validation set accuracy of 0.954
* test set accuracy of 0.934

As descrided above, I choose grayscale for inputs and used the same model structure as lenet lecture. But, this did not give good result(accurary). And then I choosed normalization filter for prepocess, and this gave better results.

When I run the model, I saw training accuracy was enough high (greater than 0.93), but validation accuracy was low (less than 0.93). Increasing epochs did not resolve this. So I thought there was overfitting for training data set.
As I described, the model difference from lenet lecture were very small, only dropouts at activations. I choose keep_prob 0.9 for training, and incresed epochs (to 15), and then validation accuracy became greater than 0.93 constantly.

This model also gave the accuracy for test set greater than 0.93, so I finished my model and parameter tune.
So, as conclusion, adding dropout and increasing epochs gave good improvement for my model.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I collected 7 German traffic signs from web.

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9] ![alt text][image10]
![alt text][image11]

I believed that these 7 images should have enough quality to classify.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


