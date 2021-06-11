# Assignment 6 

## Table of Contents:

* Contributors
* Code Explanation
* Normalization Techniques Explained
* Inferences + Graphs
* Visualization for misclassified predictions

## Contributors:

* [Ammar Adil](https://github.com/adilsammar)
* [Krithiga](https://github.com/BottleSpink)
* [Shashwat Dhanraaj](https://github.com/sdhanraaj12)
* [Srikanth Kandarp](https://github.com/Srikanth-Kandarp)

## Code Explanation:

The codebase has been modularized and we have kept the below in separate .py files

* Dataset loader 
* Model Architecture
* Data Transformations
* Backpropagation
* LR Scheduler
* Visualization
* Utils 

<To include code explanations>

[Link where the above files are available](https://github.com/adilsammar/woolly-of-cv/tree/main/assets/mnist/mnist)

The above files are used in the [Notebook](https://github.com/adilsammar/woolly-of-cv/blob/main/assets/mnist/notebook/MNIST_ALBUMENTATION_CONSOLIDATED.ipynb)

## Normalization Techniques Explained:

### Batch Normalization:
  
  What is Batch Normalization?
  Why does it work?
  What is L1?

### Layer Normalization:
  What is Layer Normalization?
  Why is it needed?

### Group Normalization:
  What is Group Normalization?
  Why is it needed inspite of layer normalization?

## Inferences + Graphs:
  
  ![Validation Losses](././assets/Validation_Losses_For_AllNorm.png)
  ![Validation Accuracy](././assets/Validation_Accuracy_For_AllNorm.png)
  
## Visualization for misclassified predictions:
 
  ![Misclassified Predictions for Batch Normalization+L1](./assets/MisPre_BNL1.png)
  ![Misclassified Predictions for Layer Normalization+L1](./assets/MisPre_LayerN.png)
  ![Misclassified Predictions for Group Normalization+L1](./assets/MisPre_GroupN.png)
  

