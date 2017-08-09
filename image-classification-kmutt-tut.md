---
layout: page
permalink: /image-classification-kmutt-tut/
---

Table of Contents:

- [Introduction](#intro)
- [Today's Goal](#goal)
- [Image Classification Pipeline](#pipeline)
    - [Input](#input)
    - [Learning](#learning)
    - [Evaluation](#evaluation)
- [Summary](#summary)
- [Additional References](#add)

<a name='intro'></a>

## Introduction
Image Classification is the task of assigning an input image one label from a fixed set of categories. This is really important in Computer Vision! If you are trying to do object detection, facial expression recognition, or even medical imaging, Image Classification is at the heart of those tasks. Today, we are going to learn and walk through creating an image classifier using Python. Because it's so early in your Machine Learning course, today's tutorial is mainly intended to be a super fast, **informal** and **fun** introduction to one of the coolest (in my opinion) areas in supervised learning.


<a name='goal'></a>

## Today's Goal
Today's goal is jump into a practical machine learning problem without worrying too much about the theory. We are going to split into three teams, based on a simple **Image Classification Pipeline**, which I will describe below. The task today is to classify images into one of four categories: **Cat, Justin Trudeau, Som Tam,** and **Temple**. Go to the *data/train_ex* folder to see examples of some of these images now.

**Format:** This is going to be an informal tutorial so most of the details will be explained as we go along in the class. Each team will be working on one of the three components of the pipeline. Each team will have tasks to complete. 

**You can complete each task by writing actual code, or just by explaining or writing down a solution.** Once you have solved a task, call me over and I will verify your answers. By the end of the class, I will compile your solutions into a functioning image classifier!

<a name='pipeline'></a>

## Image Classification Pipeline

As you've learned on your first day, a machine learning problem usually involves three main steps. First, we have to collect some observations or data. Second, our algorithm has to uncover some pattern in the data. Finally, we have to make predictions on **new** data. We often refer to these interrelated steps or tasks as a **machine learning pipeline**. The Image Classification pipeline is not too different. The data is in the form of images, each labeled with a **class** (eg. "cat"). The pattern we have to uncover is: what do each of these labels look like? Finally, we test our algorithm by giving it an image and have it predict what the correct class is.

<center>
  <img src="http://cs231n.github.io/assets/classify.png">
  <div class="figcaption"> *Source: [CS231n Classification Lecture](http://cs231n.github.io/classification/)*. "The task in Image Classification is to predict a single label (or a distribution over labels as shown here to indicate our confidence) for a given image. Images are 3-dimensional arrays of integers from 0 to 255, of size Width x Height x 3. The 3 represents the three color channels Red, Green, Blue."</div>
</center>

<div class="fig figcenter fighighlight">
  <img src="http://cs231n.github.io/assets/trainset.jpg">
  <div class="figcaption"> *Source: [CS231n Classification Lecture](http://cs231n.github.io/classification/)*. In our training set we have two pieces of information: the *image data* and *labels* telling us what class each image belongs to.</div>
</div>

<a name='input'></a>

### Task 1: Input
Our images are in two folders: *train* and *test*. The *training set* is the input data that we use to train our model. In the training folder, we also have a list of the corresponding labels for each image. The *test set* is the unseen data that we use to **evaluate** our model. Think of studying for a test. You do practice problems (training set) in order to prepare for the exam (test set). Your task is to figure out how to load these images into the variables `X_train`, `y_train`, `X_test`, and `y_test`.

#### Tasks:
1. Load the images from the folders into numpy arrays `X_train` and `X_test`.
2. Load the labels from the folders into `y_train` and `y_test`.
3. Calculate the **mean image** of the training images `X_train`. Subtract this from `X_train` and `X_test`.
4. Return the four variables in one function: `X_train`, `y_train`, `X_test`, and `y_test`.


#### Code Hints:
- Refer to the file *input.py* for guidance.
- The code for loading the training data has already been provided for you! You just need to load the test data by changing a few variable names.
- `np.mean(X, axis=0)` computes the mean of an array `X` across its rows

<a name='learning'></a>

### Task 2: Learning
Now that we have our data, we need to perform some computation on it. Essentially we are passing our image into a function or **hypothesis** that will output **scores** representing what our model predicts to be the correct correct class. We want to optimize our **parameters** in order to minimize our error on our *training set* predictions. Each time we do a prediction and update our parameters, we complete one **iteration** of training. Today, we will be creating a **Feed-forward Artificial Neural Network** with 2 hidden layers. Don't worry if you don't know what any of these terms mean. We'll have most of the code provided and explain as we go.

#### Tasks:
1. Initialize the weights for our network.
2. Write the correct matrix products to compute the output scores for our 4 classes.
3. Train our model by updating our parameters after `N` iterations (code provided). Choose an appropriate `N`.
4. Print our training accuracy to the console.
5. Store the final weights in variables for later!

#### Code Hints:
- Refer to the file *neural-net.py* for guidance.
- Can you identify some variables that we can change to affect how well our model trains?
- What does each variable affect?


<div class="fig figcenter fighighlight">
  <img src="http://cs231n.github.io/assets/imagemap.jpg">
  <div class="figcaption"> *Source: [CS231n Linear Classify Lecture](http://cs231n.github.io/linear-classify/)*. An example of mapping an image to class scores. We take in the pixels of the image as an array or vector **x** then using our weights **W** we compute the scores for each class. The highest score is our prediction. </div>
</div>

<a name='evaluation'></a>

### Task 3: Evaluation
Earlier we talked about *train* and *test* sets. In the end, what matters is your **test performance**, which is how many correct predictions your trained model makes on **new data**.

#### Tasks:
1. Write code to pass in an input `X` to your model and predict `y`.
2. Compute the number of correct predictions and divide this by the total number of `y`.
3. Print this value to the console.


## Additional References

- [CS231n First Lecture](http://cs231n.github.io/classification/) A more comprehensive lesson on Image Classification
- [CS231n Neural Networks](http://cs231n.github.io/neural-networks-1/) Introduction to neural networks
- [Deep Learning Book Ch 1](http://www.deeplearningbook.org/contents/ml.html) Textbook introduction to machine learning
