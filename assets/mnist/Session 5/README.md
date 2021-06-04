# Assignment 5 (Coding Drill Down)
### Contributers
* [Ammar Adil](https://github.com/adilsammar)
* [Krithiga](https://github.com/BottleSpink)
* [Shashwat Dhanraaj](https://github.com/sdhanraaj12)
* [Srikanth Kandarp](https://github.com/Srikanth-Kandarp)

----
## Summary 

* ## Case 1
  * ### Target 
    Basic Network
  * ### Results
    ```
    16,530 Parameters
    Best Train Accuracy: 99.26
    Best Test Accuracy: 98.92
    ```
  * ### Analysis
    Model has decent parameters but overfitting can be seen happening after epoch 8
  * ### Receptive Field Caculation
    ```
    ```
    ### [Colab Link](./Case1.ipynb)
---
* ## Case 2
  * ### Target 

     * Lighter model
     * Reduce overfitting
     * Increase Model efficiency with Batch Normalization
     * Use GAP
  * ### Results
    ```
    4,838 Parameters
    Best Train Accuracy: 98.98
    Best Test Accuracy: 98.9
    ```
  * ### Analysis

    * Model's parameters are brought down
    * Overfitting has reduced though not completely
    * Accuracy is still around 98

  * ### Receptive Field Caculation

    ```
    ```
       ### [Colab Link](./Case2.ipynb)
---
* ## Case 3
  * ### Target
    * Reduce overfitting - Augmentation
    * Learning rate optimization
    * Increase accuracy

  * ### Results
    ```
    7,038 Parameters (Changed the network)
    Best Train Accuracy: 99.14
    Best Test Accuracy: 99.5
    ```
  * ### Analysis
    * Introduced transformations like ShiftScaleRotate, RandomCrop, and RandomBrightness from albumentations library to reduce the overfitting further
    * Used LR scheduler to define a search space -> (0.01 - 0.1)

  * ### Receptive Field Caculation
    ```
    
    ```
  ### [Colab Link](./Case3.ipynb)
