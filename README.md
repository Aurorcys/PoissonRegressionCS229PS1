# Poisson Regression from Scratch

A from-scratch implementation of Poisson regression using batch gradient descent with log transform, feature normalization, and real-time training visualization. Built entirely with NumPy and pandas — no ML libraries.

## Project Structure

```
├── PoissonRegression.py   # Training + validation + animated loss plot
├── ds4_train.csv          # Training dataset
├── ds4_valid.csv          # Validation dataset
└── README.md
```

## How It Works

### Poisson Regression

Poisson regression models count data where the response variable follows a Poisson distribution. The model predicts the log of the expected count as a linear function of the inputs:

```
E[y|x] = exp(θ · x)
```

### Loss Function (Negative Log-Likelihood)

```
Loss = -mean( y · (θ · x) - exp(θ · x) )
```

### Gradient

```
∇Loss = xᵀ · (exp(θ · x) - y)
```

### Gradient Descent Update

```
θ = θ - α · ∇Loss
```

## Preprocessing

1. **Log transform** — `np.log1p()` compresses skewed features
2. **Standardization** — zero mean, unit variance
3. **Bias term** — column of ones prepended for intercept
4. **Target scaling** — divided by 1,000,000 for numerical stability

## Features

- Batch gradient descent with loss tracking
- Numerical stability via clipping (`z ∈ [-20, 20]`)
- Real-time animated loss plot during training
- Validation with deviance and explained deviance metrics

## Usage

```bash
python PoissonRegression.py
```

## Output

The script produces:

1. **Animated training loss plot** — dark theme, real-time convergence
2. **Final theta parameters**
3. **First 10 predictions vs true values**
4. **Validation deviance** — measures model fit
5. **Null deviance** — baseline (mean-only) model
6. **Explained deviance** — percentage improvement over baseline

## Example Output

```
True: [15 20 30 25 18 22 35 28 16 21]
Pred: [16.2 21.1 28.7 24.3 19.5 23.8 33.2 27.1 17.8 22.4]

Validation Deviance: 452.31
Null Deviance: 893.45
Explained Deviance: 49.38%
```

## Dependencies

```bash
pip install numpy pandas matplotlib
```

## Math Notes

Poisson regression assumes:

- Response variable is count data (non-negative integers)
- Mean = variance (equidispersion)
- Log of the rate is a linear function of predictors

The log link ensures predictions are always positive, and the Poisson distribution is the natural choice for count data.
