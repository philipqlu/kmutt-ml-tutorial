# KMUTT Machine Learning Tutorial
Image Classification workshop for Machine Learning CS532 Lecture on August 10 at KMUTT.

## Requirements
You'll need Python 2.7 or Python 3.6 to run this code. You'll also need to install libraries such as matplotlib and [PIL](http://pillow.readthedocs.io/en/4.1.x/installation.html) with `pip` or `easyinstall`.

**Update: August 10, 2017**

## Running the Code
We completed our Image Classification Pipeline together today during class. Assuming you have all the requirements installed on your system, you can run the completed version of our code in the file **run_model.py**. You will need to download and extract the **data.zip** folder in order to run this program. The rest is for you to explore and discover! Refer back to the *image-classification.md* for the full tutorial and additional references.

<center>

![img](http://i.imgur.com/Tbm9CIP.png)

<div class="figcaption"> Example of the resized, grayscale 32x32 images in our training set.
</center>
  
## Explore and Play Around
### Task 1: Input
You can change the image size, the color map, and normalization (try dividing the images by the standard deviation). You can also explore [data augmentation](https://github.com/codebox/image_augmentor) or try making a [bigger dataset](https://github.com/philipqlu/image_scrapers) with more classes.

### Task 2: Learning
You can try adding new layers to the network and playing around with the **hyperparameters** like the number of iterations, batch size, number of hidden units, and regularization.

### Task 3: Evaluation
The code so far only calculates the test accuracy. Evaluation in practice involves much more. Try computing the validation accuracy first and plotting the graphs of validation and training accuracy of your model during training. A major part of evaluation is [hyperparameter optimization](https://www.coursera.org/learn/deep-neural-network).

A lot of the inspiration and knowledge behind our research and tutorial today was gained from the awesome [CS231n Course](http://cs231n.github.io/) by Andrej Karpathy. Definitely check it out if you want to learn more about image classification and deep neural networks.


