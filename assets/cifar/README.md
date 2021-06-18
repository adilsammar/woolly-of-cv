# This article talks about training CIFAR 10 dataset


This file is submitted as part of Assignment 7 for EVA6 Course

### Table of Contents

* [Contributors](#Contributors)
* [Code Explanation](#Code-Explanation)
* [Convolution Techniques Explained](#Convolution-Techniques-Explained)
* [Transformations and Albumentations](#Transformations-and-Albumentations)
* [Graphs](#Graphs)
* [Visualization for misclassified predictions](#Visualization-for-misclassified-predictions)
* [References](#References)

### Contributors

* [Ammar Adil](https://github.com/adilsammar)
* [Krithiga](https://github.com/BottleSpink)
* [Shashwat Dhanraaj](https://github.com/sdhanraaj12)
* [Srikanth Kandarp](https://github.com/Srikanth-Kandarp)

### Code Explanation
* Model Architecture
* Visualization

### Convolution Techniques Explained

* Dilated Convolution

#### What is Dilated Convolution?  

 > In dilated convolution, we systematically aggregate multiscale contextual information without losing resolution. The dilated convolution operator has been referred to in the past as “convolution with a dilated filter”. We use the term “dilated convolution” instead of “convolution with a dilated filter” to clarify that no “dilated filter” is constructed or represented. The dilated convolution operator can apply the same filter at different ranges using different dilation factors.

#### How is it different from standard convolution?  

 > Dilated convolution is just a convolution applied to input with defined gaps. 

![Standard Convolution](FormulaStandardConvolution.png)
![Dilated Convolution](FormulaDilatedConvolution.png)

where 

> F, a discrete function,<br>
> k, filter size of (2r+1)/^2 <br>
> \*, discrete convolution operator <br>
> l, a dilation factor

> When \*l\*=1, it is standard convolution. <br>
> When \*l\*>1, it is dilated convolution. <br>

![Illustrated Dilated Convolution](IllustratedDilatedConvolution.png)


#### Why do we need dilated convolution? 

  > Dilated Convolution is specifically designed for dense prediction. Dilated convolutions supports exponential expansion of the receptive field without loss of resolution or coverage 

#### Usecases/ scenarios where dilated convolution will be of major help?

>  1. Detection of fine-details by processing inputs in higher resolutions. <br>
>  2. Broader view of the input to capture more contextual information. <br>
>  3. Faster run-time with less parameters 

Is this similar to pooling or strided convolutions? 

  > In a way, but here the output has the same size as the input. As a special case, dilated convolution with dilation 1 yields the standard convolution. 

#### Where else the dilated convolution is used?

  > Wavenet: Dilated convolutions are to increase the receptive field by orders of magnitude, without greatly increasing computational cost.</P>

Receptive Field Calculation:

  > Dilated (atrous) convolution.** Dilations introduce “holes” in a convolutional kernel. While the number of weights in the kernel is unchanged, they are no longer applied to spatially adjacent samples. Dilating a kernel by a factor of αα introduces striding of αα between the samples used when computing the convolution. This means that the spatial span of the kernel (k>0k>0) is increased to α(k−1)+1α(k−1)+1. The above derivations can be reused by simply replacing the kernel size kk by α(k−1)+1α(k−1)+1 for all layers using dilations. </P>

* Depthwise Separable Convolution

#### Problem with standard convolution:

> Traditional convolutions yield good performance, but require many computational resources. For example, one convolutional layer trained on 15x15x3 pixel images will already require more than 45000 multiplications to be made… per image!

#### Spatially separable convolutions:

>help solve this problem. They are convolutions that can be separated across their spatial axis, meaning that one large convolution (e.g. the original Conv layer) can be split into smaller ones that when convolved sequentially produce the same result. By consequence, the number of multiplications goes down, while getting the same result

The downside of these convolutions is that they cannot be used everywhere since only a minority of kernels is spatially separable. To the rescue here are **depthwise separable convolutions**. This technique simply splits convolutions differently, over a depthwise convolution and a pointwise convolution. The depthwise convolution applies the kernel to each individual channel layer only. The pointwise convolution then convolves over all channels at once, but only with a 1×1 kernel. 

Insert Depthwise Png

### Transformations and Albumentations

Lets now talk about different albumentation libraries used

#### Introduction

**Why Albumentation?** Albumentations is a Python library for image augmentations that helps boosts the performance of deep convolutional neural networks with less data. As they efficiently implements variety of image transform operations that optimized the performance of our model.

**Now lets let's see what all augumentation techniques we have used in our model in detailed,**

#### Horizontal Flip
This technique flips the input horizontally around the y-axis.
  
**Syntax**
```
class albumentations.augmentations.transforms.HorizontalFlip(always_apply=False, p=0.5)
```

Where Argumens is 

* Parameter(p) is probability of applying the transform which is by default "0". 

The final targets would be an image or Mask or Bboxes with type of the image being uint8 or float32.

#### Shift Scale Rotate
This technique randomly apply affine transforms which are translate, scale and rotate the input.

**Syntax**
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


#### Coarse Dropout
This technique helps train the rectangular regions in the image.

**Syntax** 
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

#### Grayscale
This technique helps to convert the input RGB image to grayscale. If the mean pixel value for the resulting image is greater than 127, invert the resulting grayscale image.

**Syntax**
```
class albumentations.augmentations.transforms.ToGray(always_apply=False, p=0.5)
```
Where Arguments are,
* Quality Lower is a float value and it is used to set the lower bound on the jpeg quality and the range should be [0, 100].
* Quality Upper is a float value and it is used to set the upper bound on the jpeg quality and the range should be [0, 100].

The final targets would be an Image with Type of the Image being uint8 or float32.

### Graphs

### Using Cutout

* Case 1 : 85.31 accuracy 


   ![Case_1_graph](../cifar/assets/Case_1_graph_cutout.png)

* Case 2 : 87.31 accuracy 


   ![Case_2_graph](../cifar/assets/Case_2_graph_cutout.png)

### Using Dilation

* Case 1 : 79.92 accuracy 


   ![Case_2.1_graph](../cifar/assets/Case_1_graph_dilation.png)

* Case 2 : 80.35 accuracy with Ricap
   ![Case_2.2_graph](../cifar/assets/Case_2_graph_dilation.png)

### Using Ricap 

* Case 1 : 86.26 accuracy 


   ![Case_3.1_graph](../cifar/assets/Case_1_graph_ricap.png)

* Case 2 : 87.62 accuracy 


   ![Case_3.2_graph](../cifar/assets/Case_2_graph_ricap.png)


### Visualization for misclassified predictions

### Using Cutout

* Case 1 : 85.31 accuracy 


   ![Case_1_missclassified](../cifar/assets/Case_1_missclassified_cutout.png)

* Case 1 : 85.31 accuracy 


   ![Case_1_missclassified](../cifar/assets/Case_2_missclassified_cutout.png)
   
### Using Dilation

* Case 1 : 79.92 accuracy 


   ![Case_2.1_missclassified](../cifar/assets/Case_1_missclassified_dilation.png)

* Case 2 : 80.35 accuracy using Ricap 


   ![Case_2.2_missclassified](../cifar/assets/Case_2_missclassified_dilation.png)

### Using Ricap 

* Case 1 : 86.26 accuracy 


   ![Case_3.1_missclassified](../cifar/assets/Case_1_missclassified_ricap.png)

* Case 2 : 87.62 accuracy 


   ![Case_3.2_missclassified](../cifar/assets/Case_2_missclassified_ricap.png)
   
   
### References:
Ricap: ([https://github.com/4uiiurz1/pytorch-ricap](https://github.com/4uiiurz1/pytorch-ricap))
ResidualBlocks: ([https://towardsdatascience.com/resnets-residual-blocks-deep-residual-learning-a231a0ee73d2](https://towardsdatascience.com/resnets-residual-blocks-deep-residual-learning-a231a0ee73d2))
