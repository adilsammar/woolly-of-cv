# Assignment 6 

## Table of Contents:

* Contributors
* Code Explanation
* Normalization Techniques Explained
* Inferences + Graphs
* Visualization for misclassified predictions
* References

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
  
  What is Normalization: 
  
  Input data comes in different ranges and scales. Normalization helps to change their ranges and scales to bring uniformity to data. Eg: Input images can be standardized to range of [0,255] or [0,1]. For a grayscale image, '0' being black colour while '255' being white colour. 
  
  To convert a [-500, 1000] to 0-255. 
  
  Step 1: -500 can be brought to 0 by adding 500. That brings us to [0,1500]
  Step 2: Bring [0,1500] to [0,255] -> 255/1500.
  
  Normalization can also be defined as a transformation, which ensures that the transformed data has certain statistical properties like Mean close to 0, std.dev close to 1 and so on. Normalization can be applied at different levels. Below, we will take a look at the 3 normalization techniques.
  
   ![Normalization Transformation](../../assets/NormalizationExamples.png)

### Batch Normalization:
  
  * What is Batch Normalization? 
  
  Making normalization a part of the model architecture and performing the normalization for each training mini-batch.
  
  * Why does it work? 
  
  Batch Normalization has been proved to be of help to reduce Internal Covariate Shift. 
  
  * What is Covariate Shift?
  
  The change in the distributions of layers’ inputs presents a problem because the layers need to continuously adapt to the new distribution. When the input distribution to a learning system changes, it is said to experience covariate shift.
  
   A layer with an activation function with u as the layer input, the weight matrix W and bias vector b. The model learns w,b at every backpropagation step making the gradient flowing down to u leading them to vanish and also it leads to slow convergence as the network depth increases. The nonlinear inputs not remaining stable at different parts of the training is referred to as Covariate shift. By carefully initializing and by ensurinng small learning rate could solve this problem. However, this can also be solved by making the inputs to the activation more stable. 

### Layer Normalization:
  * What is Layer Normalization?
  * Why is it needed?

### Group Normalization:
  * What is Group Normalization?
  * Do we need this inspite of layer/ batch normalization?

  
## L1:
  
## Inferences + Graphs:
  
  ![Validation Losses](../../assets/Validation_Losses_For_AllNorm.png)
  ![Validation Accuracy](../../assets/Validation_Accuracy_For_AllNorm.png)
  
## Visualization for misclassified predictions:
 
  ![Misclassified Predictions for Batch Normalization+L1](./assets/MisPre_BNL1.png)
  ![Misclassified Predictions for Layer Normalization+L1](./assets/MisPre_LayerN.png)
  ![Misclassified Predictions for Group Normalization+L1](./assets/MisPre_GroupN.png)
  
## References:
  
  * https://arxiv.org/pdf/2009.12836.pdf
  * http://proceedings.mlr.press/v37/ioffe15.pdf

