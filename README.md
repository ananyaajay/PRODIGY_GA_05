# PRODIGY_GA_05

# Neural Style Transfer

This project focuses on Neural Style Transfer. Neural Style Transfer is a technique in deep learning that applies the artistic style of one image to the content of another, producing a new, stylized image. The repository is implemented in Python and is publicly available. 


## ✓ Features:

• Implements Neural Style Transfer using PyTorch and torchvision.

• Loads and preprocesses content and style images.

• Uses a pre-trained VGG-19 model for feature extraction.

• Builds a new neural network model by inserting content and style loss layers at appropriate points.

• Optimizes the input image to match the style and content constraints using the L-BFGS optimizer.


## ✓ Concepts Used:

• Neural Style Transfer: Applying the artistic style of one image to the content of another using deep learning.

• Pre-trained Convolutional Neural Networks (VGG-19): Leveraging pre-trained models for feature extraction.

• Gram Matrix: Calculating the style representation of images using Gram matrices.

• Image Preprocessing & Postprocessing: Loading, resizing, and converting images to and from tensors.

• Using the L-BFGS optimizer to iteratively update the input image to minimize the combined content and style loss.


## ✓ Highlights:

• Applies the artistic style of one image to the content of another using neural networks.

• Separates content and style representations by defining custom loss functions (ContentLoss and StyleLoss).

• Computes style using Gram matrices to capture texture information.

• Handles image preprocessing and postprocessing for smooth operation.

• The final stylized image is saved as output.jpg after the transfer is complete.

