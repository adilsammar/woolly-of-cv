## In this article we will walk you through how to create MNIST digit dataset classifier

The target is to achieve it in under 5k params with accuracy more than 99.40

#### Moivation to keep you engaged till the end

Accuracy achieved: `99.42` within `10 epochs` with `4838` params used

Code Link: [Local NoteBook](MNIST_4838_9942.ipynb) / [CoLab Notebook](https://colab.research.google.com/drive/1uIfwHwPRwB-2jYiiTi9kksbGuU6J0B_8?usp=sharing)

### Contributer
* [Ammar Adil](https://github.com/adilsammar)
* [Krithiga](https://github.com/BottleSpink)
* [Shashwat Dhanraaj](https://github.com/sdhanraaj12)
* [Srikanth Kandarp](https://github.com/Srikanth-Kandarp)


#### Before we start Remember, `Good news will come only after your training loop ends`

![GoodThings](https://www.faxesfromuncledale.com/wp-content/uploads/Wait.gif "All Good Things to Those Who Wait")


### Basics
Before we can even start talking about machine learning, model or training we need to know what kind of problem we are trying to solve here. For us the problem in hand is classifying a digit image into its respective class (ie what digit a given image belongs to).

### Data Loader
To start with, the first thing we have to do is to load data. As we are working with `PyTorch`, there is an inbuilt functionality which helps us load data as shown below


    batch_size = 64

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=batch_size, shuffle=True)


this will load mnist data and convert it into `data loader`. DataLoader is a built in class which provides an iterator to loop over, one of the parameter this takes in is batch size which used for `minibatch` gradient descent while training.

### Data Visualization
Before we even stat with creating our model we have to look into what kinbd of data we are dealing with for this we use `matplotlib` to visualize our data. We will print some samples to see how they look.

![samples](./assets/samples.png)

As we can see from above image datasamples are all approximately centerred.

One of the other thing we need to look into our dataset is the class spread. For this we visualize our training dataset to know count of datasamples in each class.

    {0: 5923, 1: 6742, 2: 5958, 3: 6131, 4: 5842, 5: 5421, 6: 5918, 7: 6265, 8: 5851, 9: 5949}


![class_spread](./assets/class_spread.png)

From this chart we can clearly see data is evenly spread around all classes, what we can conclude from here is while training our network will not be baised to one class.

### Network Design

Designing a network is an art and an iterative process. We dont want you to go through that pain.

But before we jump into networ architecture we like to point out some of golden rules to design any network.

1. There are two kind of network designs `Smooth Design` and `Stepped Design` as shown below. In this article we have used `Stepped Design`

    <img src="assets/Pyramid-of-Khafre-Giza-Egypt.jpg" alt="drawing" width="270" height="180"/>
    <img src="assets/Step-Pyramid-of-Djoser.jpg" alt="drawing" width="270" height="180"/>

2. Try to understand what should be the right size of network you need to start with, Tip, start with like a over kill with million of params so that you know you have a good enough data to solve your problem.

3. Do not try every possible optimiztion in first iiteration. Take one step at a time.

We will now spare you with too much of `GYAN` and quickly jump on to nework design used.

![network](./assets/network.png)

This network contains block pattern as shown. We start with an image of size 1\*28\*28

1. Block 1 -> Convolution with a kernel of size 4\*3\*3 and padding 1, we do two sets of such convolution
2. Transition layer -> Here we have used MaxPolling2D to reduce channel size by half, followed by dropout of 0.01
3. Block 2 -> Convolution with a kernel of size 8\*3\*3 and padding 1, we do two sets of such convolution
4. Transition layer -> Here we have used MaxPolling2D to reduce channel size by half, followed by dropout of 0.01
5. Block 2 -> Convolution with a kernel of size 16\*3\*3 and padding 1, we do two sets of such convolution
6. Transition layer -> A 1\*1 convolution is used to reduce channel from 16 to 10
7. Output layer -> GAP is used to convert every channel to 1\*1 and then passed to softmax

    ```
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1            [-1, 4, 28, 28]              40
           BatchNorm2d-2            [-1, 4, 28, 28]               8
                Conv2d-3            [-1, 4, 28, 28]             148
           BatchNorm2d-4            [-1, 4, 28, 28]               8
             MaxPool2d-5            [-1, 4, 14, 14]               0
               Dropout-6            [-1, 4, 14, 14]               0
                Conv2d-7            [-1, 8, 14, 14]             296
           BatchNorm2d-8            [-1, 8, 14, 14]              16
                Conv2d-9            [-1, 8, 14, 14]             584
          BatchNorm2d-10            [-1, 8, 14, 14]              16
            MaxPool2d-11              [-1, 8, 7, 7]               0
              Dropout-12              [-1, 8, 7, 7]               0
               Conv2d-13             [-1, 16, 7, 7]           1,168
          BatchNorm2d-14             [-1, 16, 7, 7]              32
               Conv2d-15             [-1, 16, 7, 7]           2,320
          BatchNorm2d-16             [-1, 16, 7, 7]              32
              Dropout-17             [-1, 16, 7, 7]               0
               Conv2d-18             [-1, 10, 7, 7]             170
    AdaptiveAvgPool2d-19             [-1, 10, 1, 1]               0
    ================================================================
    Total params: 4,838
    Trainable params: 4,838
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.20
    Params size (MB): 0.02
    Estimated Total Size (MB): 0.22
    ----------------------------------------------------------------
    ```


As we can see from network summary total number of params used are `4838`


### Training




### Analysis 

We need to analyse how is our network performing. The best way to do this is to plot different parameters and see.

1. Plot learning Rate
2. Plot train loss vs test loss per epoch
3. Plot train accuracy vs test accuracy per epoch

    ![plots](./assets/plots.png)

4. Plot confusion matrix

    ![confusion_matrix](./assets/cm.png)

5. We will finally look at incorrectly classified images to see what went wrong and why

    ![incorrect](./assets/incorrect.png)