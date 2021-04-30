# Basic of Computer Vision

## Contributer
* [Ammar Adil](https://github.com/adilsammar)
* [Krithiga](https://github.com/BottleSpink)
* [Shashwat Dhanraaj](https://github.com/sdhanraaj12)
* [Srikanth Kandarp](https://github.com/Srikanth-Kandarp)

What are Channels and Kernels (according to EVA)?
Why should we (nearly) always use 3x3 kernels?
How many times do we need to perform 3x3 convolutions operations to reach close to 1x1 from 199x199 (type each layer output like 199x199 > 197x197...)
How are kernels initialized? 
What happens during the training of a DNN?
This assignment will be worth 100 points. The deadline is 7 days. 

### What are Channels and Kernels? 

![Kernel and Channel](kc.png)

In this image on the left hand side we have an input image which, this input image is pased through a process called convolution. In the process we use a matrix which is called kernel or feature exteractor or filter. The role of this matrix is to pass selective information and leave the rest. 

On the right hand side what you see an output of convolution process, which can also be called as channel or a set of channels. This output contains similar or a specific kind of information (features, shapes, texture, colour, gradient, curves, edges, patterns, objects). 

You can visualize a channel but not a kernel

### Why should we (nearly) always use 3x3 kernels?

Lets look at this image, this image shows a comparison between 3\*3 and 5\*5 kernel

![Kernel](5_5_vs_3_3.png)

1. Looking at this picture it is evidnt that we can reach at any size kernel using a 3*3 kernel
2. Using a 5x5 filter implies we need 5 * 5 = 25 unique weight parameters [i.e. we slide a single filter with 25 weights], but using two 3x3 filter → 2 * (3*3) or (9+9) unique weight parameters are needed [here, the first filter is slid with 9 weights, which creates a new layer. Then a second filter with 9 weights is slid across the new layer. Its a series of convolutions]. Therefore, as we can see while performing forward pass & back-propagation, the number of weights (used & updated) are reduced from 25 to 18. Hence a reduction in computation.
3. Nvidia GPU's which is predominately used for training DNN models are also optimized for 3*3 size kernels which gives an edge over the time and resource used in training.