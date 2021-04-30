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
Channel is a container of similar  or specific kind of information. It can be anything like features, shapes, texture, colour, gradient, curves, edges, patterns, objects
Channel is like a collection of similar type of features extracted by kernel. Kernel stores information about the image in neuron. And collection of neurons is a channel by that kernel.
Channels are can be seen

Kernel comes to use when from an image, if we want to extract some useful features of similar type.
Kernel is like a filter or like a feature extractor, kernel outputs its own channel.
Kernel extracts information and store it in neutron(collection of neutrons is a channel)
Kernels cannot be seen.

Lets put it all together in an image
