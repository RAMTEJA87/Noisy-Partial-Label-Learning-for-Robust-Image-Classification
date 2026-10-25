# CNN Training and Prediction: A Fully Worked Numerical Tutorial

**Configuration:** 20 RGB images · 10 × 10 pixels · binary
classification · 10 positive / 10 negative · 3 convolutional blocks · 3
× 3 filters · stride 2 for the first convolution · ReLU · 2 × 2 max
pooling · forward pass · binary cross-entropy · backpropagation ·
unseen-image prediction.

> **Important shape constraint and explicit assumption**
>
> If every convolution uses **valid padding**, a 3 × 3 filter, and
> stride 2, a 10 × 10 input becomes 4 × 4 after the first convolution,
> then 1 × 1 after the second, and cannot pass through a third 3 × 3
> convolution. Max pooling makes this shrinkage even more severe.
>
> To demonstrate **three sequential convolutional blocks** without
> pretending impossible dimensions work, this tutorial uses:
>
> -   **Block 1:** 3 × 3 convolution, stride 2, **valid** padding → 4 ×
>     4.
> -   **Blocks 2 and 3:** 3 × 3 convolution, stride 1, **same** padding
>     → keep 4 × 4.
> -   **Max pooling:** one 2 × 2 window, stride 2, after Block 3 → 2 ×
>     2.
>
> This is the smallest practical compromise that retains a 10 × 10 RGB
> input, a 3 × 3 first filter, stride 2, three convolutional blocks,
> ReLU, and max pooling. If **valid padding on every block** is
> non-negotiable, the input size or architecture must change.
>
> This is a transparent **toy numerical example**, not a claim of high
> real-world accuracy. To keep arithmetic inspectable, the worked
> forward/backward pass uses one filter per block and a tiny dense head.
> A production CNN would normally use many filters per block.

------------------------------------------------------------------------

## 1. What the CNN learns

A convolutional neural network learns filters (small matrices of
trainable weights) that respond to patterns in images. Early filters may
respond to edges or color changes; deeper filters combine earlier
features. A dense output layer turns the extracted features into a class
score.

Training repeatedly does four things:

1.  **Forward pass:** image → feature maps → pooled features → predicted
    probability.
2.  **Loss:** compare prediction with the correct label.
3.  **Backpropagation:** calculate how each weight contributed to the
    error.
4.  **Optimizer update:** adjust weights to reduce future loss.

The filter values below are deliberately small and hand-chosen so every
calculation can be followed. In actual training, weights are commonly
initialized randomly and learned by gradient descent.

## 2. Dataset: 20 RGB images

Each image has height 10, width 10, and 3 channels: red (R), green (G),
and blue (B). Its tensor shape is:

\[ X `\in `{=tex}`\mathbb{R}`{=tex}\^{10`\times10`{=tex}`\times3`{=tex}}
\]

Each pixel is represented by three numbers. For example, a normalized
pixel might be `(R,G,B) = (1.0, 0.0, 0.0)` for bright red. Here we use
simplified integer values 0--2 to make the arithmetic readable; divide
by 2 to normalize them to 0--1.

### 2.1 Labels

-   `y = 1`: positive class
-   `y = 0`: negative class

  Image IDs      Count            Label Illustrative visual pattern
  ----------- -------- ---------------- -----------------------------------------
  P01--P10          10                1 Red-dominant / positive-pattern images
  N01--N10          10                0 Blue-dominant / negative-pattern images
  **Total**     **20**   **10 of each** Balanced toy dataset

For a real experiment, use varied images within each class. Do not let
filenames, background, or image source trivially reveal the label.

### 2.2 Split the dataset

For a tiny demonstration, use 16 images for training and 4 for
validation. Keep both classes represented in each split.

  Split          Positive   Negative   Total
  ------------ ---------- ---------- -------
  Training              8          8      16
  Validation            2          2       4
  Total                10         10      20

A test set should ideally be separate from both training and validation.
With only 20 examples, metrics are extremely unstable; this split is
just for learning the mechanics.

## 3. One complete RGB image matrix

To keep the full arithmetic manageable, all three channels in this first
example share the same spatial pattern. The CNN still receives three
channels and the convolution sums across all three. Real RGB images do
not usually have identical channels.

### Red channel, R (10 × 10)

The first 10 × 10 channel is:

\[ R=
```{=tex}
\begin{bmatrix}
2&2&2&2&2&2&2&2&2&2\\
2&2&2&2&2&2&2&2&2&2\\
2&2&2&2&2&2&2&2&2&2\\
2&2&2&2&2&2&2&2&2&2\\
2&2&2&2&2&2&2&2&2&2\\
2&2&2&2&2&2&2&2&2&2\\
2&2&2&2&2&2&2&2&2&2\\
2&2&2&2&2&2&2&2&2&2\\
2&2&2&2&2&2&2&2&2&2\\
2&2&2&2&2&2&2&2&2&2
\end{bmatrix}
```
\]

For this one simple positive example, green and blue channels are zero:

\[ G=B=
```{=tex}
\begin{bmatrix}
0&0&0&0&0&0&0&0&0&0\\
0&0&0&0&0&0&0&0&0&0\\
0&0&0&0&0&0&0&0&0&0\\
0&0&0&0&0&0&0&0&0&0\\
0&0&0&0&0&0&0&0&0&0\\
0&0&0&0&0&0&0&0&0&0\\
0&0&0&0&0&0&0&0&0&0\\
0&0&0&0&0&0&0&0&0&0\\
0&0&0&0&0&0&0&0&0&0\\
0&0&0&0&0&0&0&0&0&0
\end{bmatrix}
```
\]

This is a deliberately simple input for arithmetic. Later images vary by
class and are not all identical copies.

## 4. Convolution: output dimensions

For a 2D convolution, output height and width are:

\[
O=`\left`{=tex}`\lfloor`{=tex}`\frac{N-F+2P}{S}`{=tex}`\right`{=tex}`\rfloor`{=tex}+1
\]

where: - (N) = input width or height - (F) = filter width or height -
(P) = padding amount - (S) = stride

### Block 1

-   Input: (10`\times10`{=tex}`\times3`{=tex})
-   Filter: (3`\times3`{=tex}`\times3`{=tex}) (the depth is 3 because
    RGB has three channels)
-   Stride: 2
-   Valid padding: (P=0)

\[
O=`\left`{=tex}`\lfloor`{=tex}`\frac{10-3}{2}`{=tex}`\right`{=tex}`\rfloor`{=tex}+1
=`\lfloor3.5`{=tex}`\rfloor`{=tex}+1=4 \]

Output feature map: **4 × 4 × 1** for one filter.

A filter spanning RGB has 3 × 3 × 3 = 27 weights plus one bias. With (K)
filters, output depth is (K).

## 5. Block 1: full convolution calculation

Use one illustrative filter. Its red-channel weights are:

\[ W_R=
```{=tex}
\begin{bmatrix}
1&0&-1\\
1&0&-1\\
1&0&-1
\end{bmatrix}
```
\]

Its green and blue weights are all zero, and bias (b=0). This filter
compares the left and right sides of a 3 × 3 red-channel patch.
(Deep-learning libraries typically implement cross-correlation, meaning
the filter is not flipped.)

For the top-left patch, rows 1--3 and columns 1--3:

\[ X\_{1:3,1:3,R}=
```{=tex}
\begin{bmatrix}
2&2&2\\
2&2&2\\
2&2&2
\end{bmatrix}
```
\]

Multiply element by element and sum:

\[ z\_{1,1} = (2`\cdot1`{=tex})+(2`\cdot0`{=tex})+(2`\cdot-1`{=tex})
+(2`\cdot1`{=tex})+(2`\cdot0`{=tex})+(2`\cdot-1`{=tex})
+(2`\cdot1`{=tex})+(2`\cdot0`{=tex})+(2`\cdot-1`{=tex})+b \]

\[ z\_{1,1}=2+0-2+2+0-2+2+0-2=0 \]

The green and blue contributions are zero because their input values and
weights are zero.

Stride 2 means the next patch starts two pixels to the right, then two
rows down. The four row-starts and column-starts are 1, 3, 5, 7. Because
this example's red channel is constant, every 3 × 3 patch produces 0.

The full feature map is therefore:

\[ Z\^{(1)}=
```{=tex}
\begin{bmatrix}
0&0&0&0\\
0&0&0&0\\
0&0&0&0\\
0&0&0&0
\end{bmatrix}
```
\]

This result is expected: the filter detects a left-right difference, and
a constant patch has no such difference. To demonstrate non-zero values,
use the following small patch example.

### 5.1 A non-zero patch example

Suppose a particular patch in a different image has red channel:

\[ P=
```{=tex}
\begin{bmatrix}
2&1&0\\
2&1&0\\
2&1&0
\end{bmatrix}
```
\]

Then:

\[ z=(2-0)+(2-0)+(2-0)=6 \]

More explicitly, each row contributes (2(1)+1(0)+0(-1)=2), so
(z=2+2+2=6). This is a raw convolution value before activation.

### 5.2 ReLU activation

ReLU is:

\[ `\operatorname{ReLU}`{=tex}(z)=`\max`{=tex}(0,z) \]

Examples: - ReLU(6) = 6 - ReLU(0) = 0 - ReLU(−3) = 0

For a feature map, ReLU is applied independently to every element. If a
pre-activation map were

\[ Z=
```{=tex}
\begin{bmatrix}
6&-2&1&0\\
-1&3&-4&2\\
0&5&-2&1\\
-3&2&0&4
\end{bmatrix}
```
\]

then

\[ A=`\operatorname{ReLU}`{=tex}(Z)=
```{=tex}
\begin{bmatrix}
6&0&1&0\\
0&3&0&2\\
0&5&0&1\\
0&2&0&4
\end{bmatrix}
```
\]

This matrix is a worked activation illustration, not the result of the
constant-image filter above.

## 6. Block 2 and Block 3

The spatial feature map after Block 1 is 4 × 4. If we applied another 3
× 3 valid convolution with stride 2, its output would be:

\[
`\left`{=tex}`\lfloor`{=tex}`\frac{4-3}{2}`{=tex}`\right`{=tex}`\rfloor`{=tex}+1=1
\]

A third 3 × 3 valid convolution cannot operate on a 1 × 1 map. Therefore
Blocks 2 and 3 use same padding and stride 1 in this tutorial.

### Block 2

-   Input: 4 × 4 × 1
-   Filter: 3 × 3 × 1
-   Stride: 1
-   Same padding: add a one-cell border for a 3 × 3 filter
-   Output: 4 × 4 × 1

A simple all-ones filter sums each 3 × 3 neighborhood. With zero
padding, an interior all-ones 3 × 3 patch produces (9); an edge patch
may contain fewer real input cells and therefore produces a smaller sum.
Then add the bias and apply ReLU.

For example, if the padded input's top-left 3 × 3 patch is:

\[
```{=tex}
\begin{bmatrix}
0&0&0\\
0&6&0\\
0&0&3
\end{bmatrix}
```
\]

and the filter is all ones, the convolution is:

\[ z=0+0+0+0+6+0+0+0+3=9 \]

If bias is (-2), (z=9-2=7), and ReLU(7) = 7.

### Block 3

-   Input: 4 × 4 × 1
-   Filter: 3 × 3 × 1
-   Stride: 1
-   Same padding
-   Output: 4 × 4 × 1

Block 3 again computes a weighted sum over each padded 3 × 3
neighborhood, adds its bias, and applies ReLU. The learned Block 3
filters can combine simpler features from Block 2 into more useful
patterns.

## 7. Max pooling

Use a 2 × 2 pooling window with stride 2. Pooling takes the maximum
value from each 2 × 2 region. It has no learned weights.

Suppose the Block 3 activation map is:

\[ A\^{(3)}=
```{=tex}
\begin{bmatrix}
1&3&2&0\\
4&2&1&5\\
0&6&7&1\\
2&3&4&8
\end{bmatrix}
```
\]

The four windows are:

Top-left: \[
```{=tex}
\begin{bmatrix}1&3\\4&2\end{bmatrix}
\quad\Rightarrow\quad \max=4
```
\]

Top-right: \[
```{=tex}
\begin{bmatrix}2&0\\1&5\end{bmatrix}
\quad\Rightarrow\quad \max=5
```
\]

Bottom-left: \[
```{=tex}
\begin{bmatrix}0&6\\2&3\end{bmatrix}
\quad\Rightarrow\quad \max=6
```
\]

Bottom-right: \[
```{=tex}
\begin{bmatrix}7&1\\4&8\end{bmatrix}
\quad\Rightarrow\quad \max=8
```
\]

Pooled result:

\[ P=
```{=tex}
\begin{bmatrix}
4&5\\
6&8
\end{bmatrix}
```
\]

Dimensions:

\[ 4`\times4`{=tex} `\xrightarrow{\text{2×2 pool, stride 2}}`{=tex}
2`\times2`{=tex} \]

## 8. Flatten and binary output layer

Flatten turns the 2 × 2 pooled map into a four-element vector:

\[ x=\[4,5,6,8\] \]

A dense neuron calculates a weighted sum plus bias:

\[ z=w_1x_1+w_2x_2+w_3x_3+w_4x_4+b \]

For an illustrative output neuron, choose:

\[ w=\[0.1,0.1,0.1,0.1\],`\quad `{=tex}b=-1 \]

Then:

\[
z=(0.1`\cdot4`{=tex})+(0.1`\cdot5`{=tex})+(0.1`\cdot6`{=tex})+(0.1`\cdot8`{=tex})-1
\] \[ z=0.4+0.5+0.6+0.8-1=1.3 \]

For binary classification, use sigmoid:

\[ p=`\sigma`{=tex}(z)=`\frac{1}{1+e^{-z}}`{=tex} \]

\[ p=`\frac{1}{1+e^{-1.3}}`{=tex}`\approx0.7858`{=tex} \]

Interpretation: the model assigns about **78.6% probability to class
1**. If the threshold is 0.5, prediction = positive. These are
illustrative dense-layer values; they are not claimed to be the trained
result for the constant RGB image in Section 3.

## 9. Binary cross-entropy loss

For label (y`\in`{=tex}{0,1}) and predicted probability (p):

\[ L=-\[y`\ln`{=tex}(p)+(1-y)`\ln`{=tex}(1-p)\] \]

### Positive example: (y=1, p=0.7858)

\[ L=-`\ln`{=tex}(0.7858)`\approx0.241`{=tex} \]

### Negative example: (y=0, p=0.7858)

\[ L=-`\ln`{=tex}(1-0.7858)=-`\ln`{=tex}(0.2142)`\approx1.541`{=tex} \]

The same prediction is reasonable for a positive example but poor for a
negative one. Training tries to reduce the loss across the batch.

For (m) images, average batch loss is:

\[
L\_{`\text{batch}`{=tex}}=`\frac{1}{m}`{=tex}`\sum`{=tex}\_{i=1}\^{m}L_i
\]

## 10. Backpropagation: output-layer gradient

For sigmoid followed by binary cross-entropy, the derivative with
respect to the output logit simplifies to:

\[ `\frac{\partial L}{\partial z}`{=tex}=p-y \]

For (p=0.7858) and (y=1):

\[ `\frac{\partial L}{\partial z}`{=tex}=0.7858-1=-0.2142 \]

For each dense weight (w_j):

\[ `\frac{\partial L}{\partial w_j}`{=tex}=(p-y)x_j \]

With (x=\[4,5,6,8\]):

-   (`\partial `{=tex}L/`\partial `{=tex}w_1=-0.2142(4)=-0.8568)
-   (`\partial `{=tex}L/`\partial `{=tex}w_2=-0.2142(5)=-1.0710)
-   (`\partial `{=tex}L/`\partial `{=tex}w_3=-0.2142(6)=-1.2852)
-   (`\partial `{=tex}L/`\partial `{=tex}w_4=-0.2142(8)=-1.7136)
-   (`\partial `{=tex}L/`\partial `{=tex}b=p-y=-0.2142)

### 10.1 SGD weight update

Gradient descent update:

\[
w\_{`\text{new}`{=tex}}=w\_{`\text{old}`{=tex}}-`\eta`{=tex}`\frac{\partial L}{\partial w}`{=tex}
\]

Choose learning rate (`\eta=0.01`{=tex}). For (w_1=0.1):

\[ w\_{1,`\text{new}`{=tex}}=0.1-0.01(-0.8568)=0.108568 \]

For the bias (b=-1):

\[ b\_{`\text{new}`{=tex}}=-1-0.01(-0.2142)=-0.997858 \]

Weights move in the direction that reduces the loss for this training
example. The next forward pass uses the updated parameters.

## 11. Gradients through ReLU, max pooling, and convolution

Backpropagation continues from the output toward earlier layers.

### 11.1 ReLU derivative

\[ `\frac{\partial\,\operatorname{ReLU}(z)}{\partial z}`{=tex}=
```{=tex}
\begin{cases}
1,&z>0\\
0,&z<0
\end{cases}
```
\]

At exactly zero, implementations choose a convention (commonly gradient
0). A negative pre-activation blocks its gradient through ReLU.

### 11.2 Max-pooling gradient

Each pooling output passes its gradient back to the location that won
the maximum. For the pooling example:

\[
```{=tex}
\begin{bmatrix}
1&3&2&0\\
4&2&1&5\\
0&6&7&1\\
2&3&4&8
\end{bmatrix}
\rightarrow
\begin{bmatrix}4&5\\6&8\end{bmatrix}
```
\]

The maxima are at positions (row 2, col 1), (row 2, col 4), (row 3, col
2), and (row 4, col 4) of the original map. If the pooled gradients are
(g\_{11},g\_{12},g\_{21},g\_{22}), those values are routed to these four
positions; other positions receive zero from this pooling path.

### 11.3 Convolution filter gradient

If an input patch is (X) and its output gradient is (`\delta`{=tex}),
each filter-weight gradient accumulates the corresponding input value
multiplied by (`\delta`{=tex}) across all output positions:

\[ `\frac{\partial L}{\partial W_{i,j,c}}`{=tex} =`\sum`{=tex}*{u,v}
X*{u+i,v+j,c},`\delta`{=tex}\_{u,v} \]

The bias gradient is:

\[
`\frac{\partial L}{\partial b}`{=tex}=`\sum`{=tex}*{u,v}`\delta`{=tex}*{u,v}
\]

The same filter is reused at many image positions, so its gradient is
the sum of contributions from every position and every training image in
the batch. For multiple channels, the calculation sums over R, G, and B
as well.

## 12. What happens for all 20 images?

The network repeats the same forward-pass and loss procedure for every
image. Here is a **schematic training ledger** showing the balanced
labels and where each example goes. The listed probabilities are
illustrative placeholders to explain loss calculation, not measured
predictions from a trained network.

  ID      Label (y)   Example probability (p) BCE loss
  ----- ----------- ------------------------- -----------------------------
  P01             1                      0.80 (-`\ln`{=tex}(0.80)=0.2231)
  P02             1                      0.75 (-`\ln`{=tex}(0.75)=0.2877)
  P03             1                      0.70 (-`\ln`{=tex}(0.70)=0.3567)
  P04             1                      0.85 (-`\ln`{=tex}(0.85)=0.1625)
  P05             1                      0.65 (-`\ln`{=tex}(0.65)=0.4308)
  P06             1                      0.90 (-`\ln`{=tex}(0.90)=0.1054)
  P07             1                      0.78 (-`\ln`{=tex}(0.78)=0.2485)
  P08             1                      0.72 (-`\ln`{=tex}(0.72)=0.3285)
  P09             1                      0.81 (-`\ln`{=tex}(0.81)=0.2107)
  P10             1                      0.68 (-`\ln`{=tex}(0.68)=0.3857)
  N01             0                      0.20 (-`\ln`{=tex}(0.80)=0.2231)
  N02             0                      0.25 (-`\ln`{=tex}(0.75)=0.2877)
  N03             0                      0.30 (-`\ln`{=tex}(0.70)=0.3567)
  N04             0                      0.15 (-`\ln`{=tex}(0.85)=0.1625)
  N05             0                      0.35 (-`\ln`{=tex}(0.65)=0.4308)
  N06             0                      0.10 (-`\ln`{=tex}(0.90)=0.1054)
  N07             0                      0.22 (-`\ln`{=tex}(0.78)=0.2485)
  N08             0                      0.28 (-`\ln`{=tex}(0.72)=0.3285)
  N09             0                      0.19 (-`\ln`{=tex}(0.81)=0.2107)
  N10             0                      0.32 (-`\ln`{=tex}(0.68)=0.3857)

The matching positive and negative probabilities above are deliberately
symmetric for demonstration. Actual probabilities come from the
network's current weights. Do not use this table as experimental
evidence.

### One example batch calculation

For the first four examples, P01--P04, the illustrative losses are:

\[ 0.2231+0.2877+0.3567+0.1625=1.0300 \]

Average loss:

\[ L\_{`\text{batch}`{=tex}}=`\frac{1.0300}{4}`{=tex}=0.2575 \]

For a real batch, calculate gradients for the batch (often averaging
them), then update weights once. Repeat for all batches.

## 13. Epochs and training loop

An **epoch** means one pass through every training image. With 16
training examples and batch size 4, there are 4 optimizer updates per
epoch.

A conceptual loop:

1.  Shuffle the 16 training examples.
2.  Take a batch of 4.
3.  Run all images through the three convolutional blocks.
4.  Pool, flatten, and calculate the sigmoid output.
5.  Compute average binary cross-entropy.
6.  Backpropagate gradients through the dense layer, pooling, ReLU, and
    convolutions.
7.  Update trainable filters and biases with the optimizer.
8.  Repeat for the remaining batches.
9.  At the end of the epoch, evaluate on the 4 validation images
    **without updating weights**.
10. Repeat for more epochs and monitor training loss and validation
    loss.

A healthy training run often shows training loss decreasing. If training
loss falls while validation loss rises, the model may be overfitting.
With only 20 images, overfitting is very likely; use more data,
augmentation where appropriate, regularization, and a separate test set
for serious evaluation.

## 14. Predict an unseen image

Suppose a new RGB image has shape 10 × 10 × 3 and was not used in
training.

1.  Apply the **same preprocessing** used for training: same channel
    order, resizing, and normalization.
2.  Run it through Block 1 (3 × 3 convolution, stride 2, valid, then
    ReLU): 10 × 10 × 3 → 4 × 4 × 1 for one filter.
3.  Run Block 2 (3 × 3 convolution, stride 1, same, then ReLU): 4 × 4 ×
    1 → 4 × 4 × 1.
4.  Run Block 3 (3 × 3 convolution, stride 1, same, then ReLU): 4 × 4 ×
    1 → 4 × 4 × 1.
5.  Max pool 2 × 2, stride 2: 4 × 4 × 1 → 2 × 2 × 1.
6.  Flatten: 2 × 2 → 4 numbers.
7.  Dense layer produces logit (z); sigmoid converts it to probability
    (p).
8.  Apply the chosen threshold, e.g. (p`\ge0.5`{=tex}) means positive,
    otherwise negative.

For example, if the unseen image produces (z=-0.7):

\[ p=`\frac{1}{1+e^{0.7}}`{=tex}`\approx0.3318`{=tex} \]

The prediction is **negative**, with model-estimated positive-class
probability about 33.2%. This number is only an illustration; the actual
probability depends on trained weights and the unseen image's pixel
values.

## 15. Summary of every layer and shape

  Stage       Operation                               Output shape
  ----------- --------------------------------------- --------------
  Input       RGB image                               10 × 10 × 3
  Block 1     3 × 3 conv, stride 2, valid, 1 filter   4 × 4 × 1
  Block 1     ReLU                                    4 × 4 × 1
  Block 2     3 × 3 conv, stride 1, same, 1 filter    4 × 4 × 1
  Block 2     ReLU                                    4 × 4 × 1
  Block 3     3 × 3 conv, stride 1, same, 1 filter    4 × 4 × 1
  Block 3     ReLU                                    4 × 4 × 1
  Pooling     2 × 2 max pool, stride 2                2 × 2 × 1
  Flatten     reshape                                 4
  Dense       one output logit                        1
  Sigmoid     probability for class 1                 1
  Threshold   e.g. 0.5                                class 0 or 1

### Parameter count in this toy network

With one filter per convolution block:

-   Block 1: (3`\times3`{=tex}`\times3`{=tex}=27) weights + 1 bias = 28
-   Block 2: (3`\times3`{=tex}`\times1`{=tex}=9) weights + 1 bias = 10
-   Block 3: (3`\times3`{=tex}`\times1`{=tex}=9) weights + 1 bias = 10
-   Dense output: 4 weights + 1 bias = 5
-   **Total = 28 + 10 + 10 + 5 = 53 trainable parameters**

Pooling and ReLU have no trainable parameters. Real CNNs commonly use
more than one filter per block.

## 16. Common misunderstandings

-   **A filter is not the same as an image.** A filter is a small set of
    learned weights applied repeatedly across an image.
-   **Stride is the movement of the filter.** Stride 2 moves two pixels
    between convolution positions.
-   **RGB filters have depth 3.** A 3 × 3 filter for RGB input is really
    3 × 3 × 3.
-   **ReLU does not learn weights.** It changes negative values to zero.
-   **Max pooling does not learn weights.** It retains the maximum
    within each window.
-   **Loss is not accuracy.** Loss measures prediction error; accuracy
    measures the fraction of correct classifications at a threshold.
-   **Training is not the same as prediction.** Training updates
    parameters; prediction uses the learned parameters without updating
    them.
-   **More layers are not automatically better.** Spatial dimensions
    must remain large enough for the selected filters.
-   **Twenty images are not enough for reliable generalization.** This
    dataset is for hand calculation and understanding, not a robust
    deployed classifier.

## 17. Final mental model

\[ `\boxed{
\text{RGB image}
\rightarrow\text{Convolution}
\rightarrow\text{ReLU}
\rightarrow\text{Convolution blocks}
\rightarrow\text{Max pool}
\rightarrow\text{Flatten}
\rightarrow\text{Sigmoid}
\rightarrow\text{Loss}
}`{=tex} \]

During training, backpropagation sends the loss signal backward to
update the filters and dense weights. During prediction, the model only
performs the forward pass and returns a probability and class decision.
