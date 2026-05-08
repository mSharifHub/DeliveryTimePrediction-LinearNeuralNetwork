# Delivery Time Prediction — Linear Neural Network

A simple neural network built with PyTorch that predicts delivery time (minutes) based on distance (miles).

---

## Model Architecture

| Layer | Type | Input | Output |
|-------|------|-------|--------|
| 1 | Linear | 1 | 1 |

A single `nn.Linear(1, 1)` layer — equivalent to learning the equation:

```
time = weight × distance + bias
```

---

## Dataset

- **60 data points** covering distances from 0.3 to 15.0 miles
- **Realistic variance** baked in to simulate real-world conditions:
  - Restaurant wait times
  - Rush hour traffic
  - Highway vs city street routes
  - Parking and last-mile difficulty
- **Base pattern**: `time ≈ 5 × distance + 2 minutes`
- **Variance**: ±20–30% around the base pattern

---

## Training

| Parameter | Value |
|-----------|-------|
| Loss function | MSE (Mean Squared Error) |
| Optimizer | SGD |
| Learning rate | 0.01 |
| Max epochs | 1000 |
| Early stopping patience | 50 epochs |

**Early stopping** halts training when the loss improvement falls below `1e-4` for 50 consecutive epochs, preventing unnecessary computation.

The final loss will not reach zero — a residual of **~8–15 MSE** is expected and correct, representing the irreducible error from real-world noise.

---

## Key Concepts

### `torch.no_grad()`
Used during prediction and plotting. Disables gradient tracking since no backpropagation is needed, reducing memory usage and speeding up inference.

### Early Stopping
Monitors loss improvement each epoch. If no meaningful improvement occurs for `patience` epochs, training stops automatically.

### Irreducible Error
Because the data has intentional variance (same distance, different times), a linear model cannot achieve zero loss. This is realistic — no model can perfectly predict delivery time.

---

## Usage

Run the notebook cells in order:

1. **Import libraries** — PyTorch, Matplotlib
2. **Define dataset** — distances and times tensors
3. **Define model** — `nn.Sequential(nn.Linear(1, 1))`
4. **Train** — runs up to 1000 epochs with early stopping
5. **Plot** — visualizes actual vs predicted delivery times
6. **Predict** — enter any distance in miles to get a predicted time

```
Enter distance in miles: 7.5
Predicted delivery time for 7.5 miles: 39.42 minutes
```

---

## Limitations

- Only one input feature (distance). Real delivery time also depends on:
  - Time of day
  - Weather
  - Traffic conditions
  - Restaurant prep time
- A more accurate model would use multiple input features (see `nn.Linear(n, 1)`)
# DeliveryTimePrediction-LinearNeuralNetwork
