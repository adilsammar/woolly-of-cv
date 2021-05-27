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

1. There are two kind of network designs `Smooth Design` and `Stepped Design` as shown below

    <img src="assets/Pyramid-of-Khafre-Giza-Egypt.jpg" alt="drawing" width="270" height="180"/>
    <img src="assets/Step-Pyramid-of-Djoser.jpg" alt="drawing" width="270" height="180"/>



