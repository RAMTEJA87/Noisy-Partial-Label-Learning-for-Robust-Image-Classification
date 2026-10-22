# Convolutional Neural Network (CNN): Complete Beginner's Guide

## 1. What is a CNN?

A **Convolutional Neural Network (CNN)** is a deep-learning model
designed to learn patterns from grid-like data, especially images. It is
widely used for image classification, object detection, face
recognition, and medical-image analysis.

For example, suppose we want to build a model that classifies an image
as either **Cat** or **Dog**. A CNN learns useful visual patterns from
training images and uses those patterns to predict the class of a new
image.

A CNN does not normally receive an image as a word or a label. It
receives numeric pixel values.

## 2. The complete CNN pipeline

A common beginner-friendly CNN pipeline is:

1.  Input image
2.  Preprocessing
3.  Convolution
4.  Activation (usually ReLU)
5.  Pooling
6.  Repeat convolution, activation, and pooling
7.  Flatten or Global Average Pooling
8.  Fully connected (Dense) layer
9.  Output layer
10. Loss calculation and backpropagation during training

Not every CNN uses exactly these layers. Modern architectures may use
normalization, residual connections, or global average pooling.

## 3. Phase 1 --- Input image

The input layer receives the image as an array of pixel values.

For a color image with height 128 pixels and width 128 pixels, the shape
is:

`128 × 128 × 3`

The three channels represent Red, Green, and Blue (RGB).

-   Height: 128 pixels
-   Width: 128 pixels
-   Channels: 3

A grayscale image has one channel, so its shape could be
`128 × 128 × 1`.

**Example:** A cat image is converted into numbers representing the
brightness of red, green, and blue at each pixel.

## 4. Phase 2 --- Preprocessing

Before training, images are usually resized to a consistent size and
pixel values are normalized.

For example, an 8-bit pixel channel normally has a value from 0 to 255.
Dividing by 255 converts it to the range 0 to 1:

`normalized_pixel = pixel / 255.0`

Examples:

-   0 becomes 0.0
-   127.5 becomes approximately 0.5
-   255 becomes 1.0

Consistent image dimensions let images be processed in batches.
Normalization can make optimization easier.

Other preprocessing choices can include data augmentation, such as
random flips, small rotations, or crops. Augmentation should preserve
the correct label.

## 5. Phase 3 --- Convolution layer (Conv2D)

The convolution layer learns visual features by applying small matrices
called **filters** or **kernels** across an image or feature map.

A filter computes responses to patterns such as edges, corners,
textures, and shapes. During training, the model learns the filter
values.

A common kernel size is `3 × 3`.

### Simple numerical example

Suppose a filter produces this small output region:

``` text
Input feature region:      Filter:
1  2  0                    1  0 -1
3  1  2                    1  0 -1
0  1  3                    1  0 -1
```

A convolution operation multiplies corresponding entries and sums the
results (implementation details may include padding and channel
handling). The output value indicates how strongly the filter responds
at that position.

### Important convolution parameters

-   **Filters:** Number of patterns the layer can learn. For example,
    `32` filters produce 32 output feature maps.
-   **Kernel size:** Spatial size of each filter, such as `3 × 3`.
-   **Stride:** Number of pixels the filter moves each step. A stride of
    1 moves one pixel at a time.
-   **Padding:** Extra border pixels added around the input. `same`
    padding is often used to preserve spatial dimensions when stride is
    1; `valid` uses no padding.

### Output-size formula

For one spatial dimension:

`output_size = floor((input_size + 2 × padding - kernel_size) / stride) + 1`

For a 128-pixel input, a 3-pixel kernel, no padding, and stride 1:

`(128 - 3) / 1 + 1 = 126`

So that dimension becomes 126.

## 6. Phase 4 --- Activation function (ReLU)

After convolution, a CNN commonly applies ReLU, which stands for
**Rectified Linear Unit**.

Formula:

`ReLU(x) = max(0, x)`

Examples:

    Input   ReLU output
  ------- -------------
       -5             0
       -2             0
        0             0
        3             3
        7             7

ReLU turns negative values into zero and leaves positive values
unchanged.

**Why use it?** It introduces non-linearity, allowing the network to
learn complex patterns. Without nonlinear activations, stacking only
linear operations would still behave like a linear transformation.

## 7. Phase 5 --- Pooling layer

Pooling reduces the height and width of feature maps. This reduces
computation and can provide some tolerance to small shifts in features.

### Max pooling

Max pooling selects the largest value in each region.

Example using a `2 × 2` window:

``` text
1  3
2  4
```

The maximum is `4`, so this region produces the value 4.

For a larger feature map, the pooling window slides across it and
creates a smaller map.

### Average pooling

Average pooling calculates the average value in each region. For the
same values:

`(1 + 3 + 2 + 4) / 4 = 2.5`

A common setup is a `2 × 2` window with stride 2, which typically halves
the feature map's height and width.

Pooling is common, but not mandatory in every CNN architecture.

## 8. Phase 6 --- Repeat feature extraction

CNNs often stack multiple convolution blocks. Early layers tend to learn
simple patterns, while deeper layers can combine them into more complex
patterns.

-   **Early layers:** Edges, lines, and simple textures.
-   **Middle layers:** Corners, textures, eyes, ears, or other object
    parts.
-   **Deeper layers:** Combinations of features that help represent
    objects and categories.

These are general tendencies, not hard-coded rules. The network learns
the filters from data.

Example architecture progression:

``` text
Image: 128 × 128 × 3
       ↓
Conv2D: 32 filters
       ↓
ReLU
       ↓
MaxPooling: 2 × 2
       ↓
Conv2D: 64 filters
       ↓
ReLU
       ↓
MaxPooling: 2 × 2
       ↓
Conv2D: 128 filters
       ↓
ReLU
       ↓
MaxPooling: 2 × 2
```

The number of channels increases from 32 to 64 to 128, while pooling
reduces the spatial dimensions.

## 9. Phase 7 --- Flatten layer

After convolution and pooling, the output is a multi-dimensional
collection of feature maps. A **Flatten** layer reshapes it into one
long vector for a Dense layer.

For example:

`7 × 7 × 64 = 3,136 values`

Flatten converts a `7 × 7 × 64` tensor into a vector of 3,136 values.

Flatten does not learn features or classify the image. It changes the
shape of the data.

### Alternative: Global Average Pooling

Instead of flattening every spatial value, Global Average Pooling
averages each feature map into one value. For example, a `7 × 7 × 64`
tensor becomes a vector of 64 values. This can greatly reduce the number
of parameters in the classifier.

## 10. Phase 8 --- Fully connected (Dense) layer

A Dense layer connects each neuron to all outputs from the previous
layer. It combines the learned visual features to help make a decision.

A Dense layer commonly computes:

`y = activation(Wx + b)`

Where:

-   `x` is the input feature vector.
-   `W` contains learned weights.
-   `b` is a learned bias.
-   `activation` is a function such as ReLU.

For a cat-versus-dog model, the Dense layer can combine information from
learned features such as ear shape, fur texture, and muzzle shape.

## 11. Phase 9 --- Dropout (optional)

Dropout is a regularization technique. During training, it randomly sets
a portion of activations to zero. For example, `Dropout(0.5)` uses a
dropout rate of 50%.

This can help reduce overfitting, where a model learns training examples
too specifically and performs poorly on new images.

Dropout is active during training but is disabled during normal
inference. It is optional, not a required CNN layer.

## 12. Phase 10 --- Output layer

The output layer produces scores or probabilities for the target
classes.

### Binary classification: Cat or Dog

A common approach uses one output neuron with sigmoid activation:

`sigmoid(z) = 1 / (1 + exp(-z))`

The output lies between 0 and 1 and can be interpreted as the model's
estimated probability for the positive class (for example, Dog),
assuming the training setup defines Dog as the positive class.

Example:

-   Output 0.90: estimated probability for Dog is 90%.
-   Output 0.15: estimated probability for Dog is 15%.

A threshold such as 0.5 can be used to turn the probability into a
class, but the threshold may be adjusted for the application.

### Multi-class classification

For categories such as Cat, Dog, Horse, and Bird, a common output layer
has four neurons and uses **softmax**.

Softmax converts class scores into values that sum to 1. The class with
the highest probability is often selected as the prediction.

## 13. Full worked example --- Cat versus Dog

Imagine a dataset containing labeled cat and dog images.

### Step A: Prepare the data

1.  Collect images with correct labels.
2.  Split the data into training, validation, and test sets.
3.  Resize each image to `128 × 128`.
4.  Normalize pixel values to the range 0--1.
5.  Create batches of images and labels.

Keep the test set separate from training and model-selection decisions.

### Step B: Pass one image through the CNN

Suppose the input is a cat image with shape `128 × 128 × 3`.

1.  **Convolution:** Filters respond to visual patterns.
2.  **ReLU:** Negative activations become zero.
3.  **Pooling:** Feature maps become smaller.
4.  **More convolution blocks:** The network learns more complex
    features.
5.  **Flatten:** Feature maps become a vector.
6.  **Dense layer:** Features are combined.
7.  **Sigmoid output:** The model estimates the probability that the
    image is a Dog.

If the model outputs `0.08` for Dog, it estimates `0.92` for Cat in a
two-class setup where those are the only classes and the output
represents Dog probability. The predicted class would be Cat under a 0.5
threshold.

### Step C: Train the model

During training, the model compares predictions with the correct labels
using a loss function. For binary classification, binary cross-entropy
is commonly used.

Training loop concept:

1.  Forward pass: calculate predictions.
2.  Loss calculation: measure prediction error.
3.  Backpropagation: calculate gradients of the loss.
4.  Optimizer update: adjust weights and biases.
5.  Repeat for batches and epochs.

An **epoch** is one complete pass through the training dataset.

### Step D: Evaluate the model

Use validation data to monitor performance and help choose model
settings. After decisions are finalized, evaluate once on the held-out
test set.

Useful metrics include accuracy, precision, recall, F1-score, and a
confusion matrix. For imbalanced datasets, accuracy alone may be
misleading.

## 14. Complete working code (TensorFlow/Keras)

This example assumes you have two folders, `train` and `validation`,
each containing class subfolders named `cat` and `dog`.

Expected folder structure:

``` text
dataset/
├── train/
│   ├── cat/
│   └── dog/
└── validation/
    ├── cat/
    └── dog/
```

Install TensorFlow if needed:

``` bash
pip install tensorflow
```

Python code:

``` python
import tensorflow as tf
from tensorflow.keras import layers, models

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/validation",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

print("Class names:", train_ds.class_names)

# Normalize pixel values from 0–255 to 0–1.
normalization = layers.Rescaling(1.0 / 255)

model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    normalization,

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    # Binary output: probability for the positive class.
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

test_loss, test_accuracy = model.evaluate(val_ds)
print("Evaluation loss:", test_loss)
print("Evaluation accuracy:", test_accuracy)

# Save the trained model.
model.save("cat_dog_cnn.keras")
```

**Important:** The folder labels determine the class mapping. Check
`train_ds.class_names` so you know which class corresponds to the
positive label used by the binary output. This example evaluates on the
validation folder only because a separate test folder is not included;
for a final unbiased evaluation, keep a separate test set.

### Predict a single image

``` python
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("cat_dog_cnn.keras")

image_path = "sample_cat.jpg"

image = tf.keras.utils.load_img(
    image_path,
    target_size=(128, 128)
)

image_array = tf.keras.utils.img_to_array(image)
image_array = np.expand_dims(image_array, axis=0)

# The model includes Rescaling, so pass pixel values in 0–255.
dog_probability = float(model.predict(image_array, verbose=0)[0][0])

print("Estimated Dog probability:", dog_probability)
print("Estimated Cat probability:", 1.0 - dog_probability)

if dog_probability >= 0.5:
    print("Prediction: Dog")
else:
    print("Prediction: Cat")
```

The prediction code assumes the positive class is Dog. Confirm the
class-index mapping from the dataset before using that assumption.

## 15. Training versus inference

### Training

-   Images and labels are provided.
-   The model makes predictions.
-   The loss function measures error.
-   Backpropagation calculates gradients.
-   The optimizer updates weights and biases.
-   This repeats over many batches and epochs.

### Inference (prediction)

-   A new image is provided.
-   The image is preprocessed in the same way as training images.
-   The trained model performs a forward pass.
-   The output is interpreted as a prediction.
-   Weights are not updated.

## 16. Important CNN interview questions

**1. What is a CNN?**\
A neural network designed to learn spatial patterns in images using
operations such as convolution.

**2. What is a filter or kernel?**\
A learnable small matrix used to compute responses to patterns in an
input.

**3. Why is ReLU used?**\
It introduces non-linearity and is computationally simple.

**4. Why is pooling used?**\
It reduces spatial dimensions and computation and can provide some
tolerance to small spatial shifts.

**5. What is the difference between convolution and pooling?**\
Convolution applies learned filters; pooling summarizes values in local
regions and usually has no learned weights.

**6. What does Flatten do?**\
It reshapes multidimensional feature maps into a one-dimensional vector.

**7. What is overfitting?**\
When a model fits the training data too closely and generalizes poorly
to unseen data.

**8. How can overfitting be reduced?**\
Use more suitable training data, data augmentation, dropout, weight
regularization, early stopping, or a simpler model.

**9. What is backpropagation?**\
The procedure for calculating gradients of the loss with respect to
model parameters.

**10. What is the difference between an epoch and a batch?**\
A batch is a subset of examples processed together. An epoch is one
complete pass through the training dataset.

**11. What is the difference between sigmoid and softmax?**\
Sigmoid is commonly used for a single binary output or independent
labels. Softmax is commonly used to represent a probability distribution
across mutually exclusive classes.

**12. Does every CNN need Flatten?**\
No. Global Average Pooling or other architectural choices can replace
it.

## 17. Quick revision table

  Component         Main purpose
  ----------------- ----------------------------------------
  Input             Receives pixel values
  Preprocessing     Resizes and normalizes images
  Convolution       Learns visual features
  ReLU              Adds non-linearity
  Pooling           Reduces spatial dimensions
  Flatten           Reshapes feature maps into a vector
  Dense             Combines features
  Dropout           Regularization during training
  Output            Produces class scores or probabilities
  Loss              Measures prediction error
  Backpropagation   Calculates gradients
  Optimizer         Updates parameters

## 18. One-line summary

A CNN receives an image, extracts increasingly complex features using
convolution and activation layers, reduces spatial dimensions using
pooling, combines the learned features in a classifier, and produces a
prediction. During training, the loss, backpropagation, and optimizer
work together to improve that prediction.
