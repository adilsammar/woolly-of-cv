# This article talks about training CIFAR 10 dataset


This file is submitted as part of Assignment 7 for EVA6 Course

## Table of Contents

* [Contributors](#Contributors)
* [Code Explanation](#Code-Explanation)
* [Convolution Techniques Explained](#Convolution-Techniques-Explained)
* [Transformations and Albumentations](#Transformations-and-Albumentations)
* [Graphs](#Graphs)
* [Visualization for misclassified predictions](#Visualization-for-misclassified-predictions)
* [References](#References)

## Contributors

* [Ammar Adil](https://github.com/adilsammar)
* [Krithiga](https://github.com/BottleSpink)
* [Shashwat Dhanraaj](https://github.com/sdhanraaj12)
* [Srikanth Kandarp](https://github.com/Srikanth-Kandarp)

## Code Explanation
* Model Architecture
* Visualization

## Convolution Techniques Explained:

* Dilated Convolution
* Depthwise Separable Convolution

## Transformations and Albumentations:
Lets now talk about different albumentation libraries used,

### Introduction 
  - ### Why Albumentation ? 

    Albumentations is a Python library for image augmentations that helps boosts the performance of deep convolutional neural networks with less data. As they efficiently implements variety of image transform operations that optimized the performance of our model.

### Now lets let's see what all albumentation techniques we have used in our model in detailed,

* ### Horizontal Flip
  ### This technique flips the input horizontally around the y-axis.
  
  ### Syntax 
    ```
    class albumentations.augmentations.transforms.HorizontalFlip(always_apply=False, p=0.5)
    ```
    
    * Where Argumens is Parameter(p) is probability of applying the transform which is by default "0". The final targets would be an image or Mask or Bboxes with type of the image being uint8 or float32.

* ### Shift Scale Rotate
    ### This technique randomly apply affine transforms which are translate, scale and rotate the input.
    ### Syntax 
    ```
    class albumentations.augmentations.transforms.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=1, border_mode=4, always_apply=False, p=0.5)

    ```
    Where Arguments are,

     * Shift limit is a (float,float) or can also be a single float value which is the shift factor range for both height and width. If shift_limit is a single float value, the range will be (-shift_limit, shift_limit). Absolute values for lower and upper bounds should lie in range [0, 1] and default being "0.0625".


     * Scale limit is a (float, float) or can also be a single float value which is the scaling factor range. If scale_limit is a single float value, the range will be (-scale_limit, scale_limit) and default being "0.1".


     * Rotate limit is a (int, int) or can also be a single int value which is the rotation range. If rotate_limit is a single int value, the range will be (-rotate_limit, rotate_limit)and default being "45".


     * Interpolation is a flag that is used to specify the interpolation algorithm and that can be one of these all below and default being "INTER_LINEAR",
          1. INTER_NEAREST,
          2. INTER_LINEAR
          3. INTER_CUBIC
          4. INTER_AREA
          5. INTER_LANCZOS4
       


    * Border Mode is a flag that is used to specify the pixel extrapolation method and that can be one of these all below and default being "BORDER_REFLECT_101",
  
         1. BORDER_REFLECT_101
         2. BORDER_REPLICATE
         3. BORDER_REFLECT
         4. BORDER_WRAP
         5. BORDER_REFLECT_101
        

    * Just like for Horizontal Flip Parameter(p) is probability of applying the transform which is by default "0" .


    The final targets would be an Image or Mask with Type of the Image being uint8 or float32.


* ### Coarse Dropout
  ### This technique helps train the rectangular regions in the image.

  ### Syntax 
    ```
    class albumentations.augmentations.transforms.CoarseDropout (max_holes=8, max_height=8, max_width=8, min_holes=null, min_height=null, min_width=null, fill_value=0, mask_fill_value=null, always_apply=False, p=0.5)
    ```
    Where Arguments are,

    * Max Holes is a int value and it is used to declare the maximum number of regions to zero out.


    * Max Height is a int Value and it is used to set the maximum height of the hole.


    * Max Width is a int Value and it is used to set the maximum width of the hole.

    * Min Holes is a int value and it is used to declare minimum number of regions to zero out and if the value is null then Min Holes is be set to  value of max_holes if not then default is null.

    * Min Height is a int value and it is used to set the minimum height of the hole and if the value is null then Max Height is be set to value of Max Height if not then default is null.


    * Min Width is a int value and it is used to set the minimum height of the hole and if the value is null then Max Width is be set to value of Max Width if not then default is null .


    * Fill Value can be a int,float of list of int,float values that can be used to defind the dropped pixels.


    * Mask Fill Value can be be a int,float of list of int,float values thae can be used to fill the value for dropped pixels in mask if the value is null then the mask won't be affected by default it is null


    The final targets would be an Image or Mask with Type of the Image being uint8 or float32.

* ### Grayscale
  ### This technique helps to convert the input RGB image to grayscale. If the mean pixel value for the resulting image is greater than 127, invert the resulting grayscale image.

  ### Syntax 
    ```
    class albumentations.augmentations.transforms.ToGray(always_apply=False, p=0.5)
    ```
    Where Arguments are,
     * Quality Lower is a float value and it is used to set the lower bound on the jpeg quality and the range should be [0, 100].


     * Quality Upper is a float value and it is used to set the upper bound on the jpeg quality and the range should be [0, 100].

    The final targets would be an Image with Type of the Image being uint8 or float32.

## Graphs

## Visualization for misclassified predictions
  
## References:
Ricap: ([https://github.com/4uiiurz1/pytorch-ricap](https://github.com/4uiiurz1/pytorch-ricap))