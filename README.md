# From-Scratch Implementation of a Multilayer Neural Network

A fully manual implementation of a Multilayer Perceptron (MLP) trained on MNIST, no PyTorch, no TensorFlow, no autograd. Every forward pass, every gradient, every weight update written explicitly in NumPy.

---

## Why build it from scratch?

Frameworks are great for production. But if you use `model.fit()` you wont understand what's happening underneath.

By implementing backpropagation manually, you're forced to confront what the math actually means.

---

## The Dataset: MNIST

MNIST is a classic benchmark: 70,000 grayscale images of handwritten digits (0–9), each 28×28 pixels = **784 input features**.

**Preprocessing:** pixel values are normalized from `[0, 255]` to `[-1, 1]` using:

```
X = ((X / 255.) - 0.5) * 2
```

This is **not** the same as standardization or min-max scaling you'd do for tabular data. Here we're scaling uniformly because all pixels share the same scale and meaning, we just want the network to see values centered around zero, which helps gradients flow better during training.

**Splits:**
- Training: 55,000 samples
- Validation: 5,000 samples
- Test: 10,000 samples

Stratified splitting ensures each class is proportionally represented in every split.
<img width="640" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/11f7767d-03a2-4785-98e0-bcff77def23d" />

---

## Architecture

```
Input (784)  →  Hidden Layer (50, sigmoid)  →  Output Layer (10, sigmoid)
```
<img width="655" height="462" alt="image" src="https://github.com/user-attachments/assets/43046cc1-a9e4-4d9f-9fd7-a76e1597e94e" />

A simple two-layer MLP. The hidden layer learns intermediate representations; the output layer produces 10 probability-like scores, one per digit class.

**Why sigmoid?**  
Sigmoid squashes any value into `(0, 1)`, which pairs naturally with MSE loss and one-hot targets. It also has a clean derivative: `σ(z) * (1 - σ(z))`  which shows up directly in the backprop equations below.

**Weights** are initialized from a normal distribution (mean=0, std=0.1). Biases start at zero.

---

## Forward Pass

```
z_h   = X · W_h.T + b_h        # hidden pre-activation
a_h   = sigmoid(z_h)            # hidden activation

z_out = a_h · W_out.T + b_out  # output pre-activation  
a_out = sigmoid(z_out)          # output activation (predictions)
```

Each layer is just a linear transformation followed by a non-linearity. That's it.

---

## Loss Function: Mean Squared Error

```
L = mean((y_onehot - a_out)²)
```

Targets are one-hot encoded — e.g. digit `3` becomes `[0,0,0,1,0,0,0,0,0,0]`. The network's output is compared to this vector element-wise.

---

## Backward Pass (Backpropagation)

This is the core of the project. Backprop is just the **chain rule of calculus**, applied layer by layer from the output back to the input.

### Output Layer Gradients
<img width="717" height="393" alt="image" src="https://github.com/user-attachments/assets/c07dc00c-a70f-4c8a-ad38-0ed7a2d9ebb7" />
<img width="671" height="568" alt="image" src="https://github.com/user-attachments/assets/c38afdec-7dbe-44fd-8213-64018b083782" />

```
dL/da_out  = 2*(a_out - y_onehot) / n        # MSE derivative
da_out/dz  = a_out * (1 - a_out)             # sigmoid derivative
delta_out  = dL/da_out * da_out/dz           # combined error signal

dL/dW_out  = delta_out.T · a_h
dL/db_out  = sum(delta_out, axis=0)
```

### Hidden Layer Gradients
<img width="649" height="603" alt="image" src="https://github.com/user-attachments/assets/f0b76870-d499-449d-8232-ceca6bf97f27" />

```
dL/da_h   = delta_out · W_out                # error propagated back
da_h/dz_h = a_h * (1 - a_h)                 # sigmoid derivative
delta_h   = dL/da_h * da_h/dz_h

dL/dW_h   = delta_h.T · X
dL/db_h   = sum(delta_h, axis=0)
```

---

## Training Loop

**Mini-batch Stochastic Gradient Descent** with batch size = 100.

```
for each epoch:
    shuffle training data
    for each mini-batch:
        forward pass → compute a_h, a_out
        backward pass → compute all gradients
        update weights:
            W -= lr * dL/dW
            b -= lr * dL/db
```

Mini-batches give a good balance: less noisy than pure SGD (batch size 1), faster than full-batch GD. Shuffling before each epoch prevents the model from memorizing order.

**Hyperparameters:**
| Parameter | Value |
|---|---|
| Epochs | 50 |
| Mini-batch size | 100 |
| Learning rate | 0.1 |
| Hidden units | 50 |

---

## Results

After 50 epochs:

- **Training accuracy:** ~95.6%
- **Validation accuracy:** ~94.8%
- **Test accuracy:** ~94.5%

<img width="640" height="480" alt="Figure_2" src="https://github.com/user-attachments/assets/64ddb32f-7208-4d49-ac65-3541d114a4bd" />
<img width="640" height="480" alt="Figure_3" src="https://github.com/user-attachments/assets/c4653574-843e-4f25-8706-563b4632f67b" />

The loss curve decreases smoothly, and training/validation accuracy track closely so there is no significant overfitting, little overfitting start after around epoch 25

---



## What I Learned

The most important realization: **backpropagation is a chain rule**, applied carefully, with attention to matrix dimensions at every step. Once you work through it once by hand, the "black box" disappears entirely.
