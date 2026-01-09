"""
Model Fitting with Linear Regression and Poisson GLM

aim: 1) Fit a linear regression model to predict a continuous target.
     2) Fit a Poisson GLM to predict spike counts from a stimulus feature.

Why this matters: These are core tools for building and interpreting simple encoding models in neuroscience.

Source: Neuromatch Academy – Machine Learning / Model Fitting
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.metrics import r2_score, mean_squared_error


def run():
    rng = np.random.default_rng(0)

    
    #Part 1: Linear Regression
    
    n_samples = 800
    n_features = 6

    #Create synthetic features
    X = rng.normal(0, 1, size=(n_samples, n_features))

    #True weights (ground truth)
    w_true = rng.normal(0, 1, size=n_features)

    # Continuous target with noise
    noise = rng.normal(0, 1.0, size=n_samples)
    y = X @ w_true + noise

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    scaler = StandardScaler()
    X_train_z = scaler.fit_transform(X_train)
    X_test_z = scaler.transform(X_test)

    lin = LinearRegression()
    lin.fit(X_train_z, y_train)
    y_pred = lin.predict(X_test_z)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"[Linear Regression] R^2: {r2:.3f} | MSE: {mse:.3f}")

    # Plot predicted vs true
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.xlabel("True y")
    plt.ylabel("Predicted y")
    plt.title(f"Linear Regression: Predicted vs True (R^2 = {r2:.3f})")
    plt.tight_layout()
    plt.savefig("linear_regression_pred_vs_true.png", dpi=300)
    plt.show()

    
    # Part 2: Poisson GLM (spike counts)

    #We'll simulate a single stimulus feature 
    n_trials = 1200
    stimulus = rng.normal(0, 1, size=n_trials)

    # True encoding: firing rate lambda = exp(b0 + b1 * stimulus)
    b0_true = 1.2
    b1_true = 0.9
    lam = np.exp(b0_true + b1_true * stimulus)  # expected spike count per trial

    # Generate spike counts from Poisson
    spikes = rng.poisson(lam)

    X_glm = stimulus.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(
        X_glm, spikes, test_size=0.25, random_state=0
    )

    #PoissonRegressor is a GLM with log link (standard for spike counts)
    glm = PoissonRegressor(alpha=0.0, max_iter=1000)
    glm.fit(X_train, y_train)
    y_pred_counts = glm.predict(X_test)

    #A simple, readable metric here: correlation between true and predicted spike counts
    corr = np.corrcoef(y_test, y_pred_counts)[0, 1]
    print(f"[Poisson GLM] Corr(true, predicted): {corr:.3f}")

    # Plot predicted mean counts vs stimulus (smooth curve)
    stim_grid = np.linspace(stimulus.min(), stimulus.max(), 200).reshape(-1, 1)
    pred_rate = glm.predict(stim_grid)

    plt.figure()
    plt.scatter(stimulus, spikes, alpha=0.25)
    plt.plot(stim_grid.ravel(), pred_rate, linewidth=2)
    plt.xlabel("Stimulus feature")
    plt.ylabel("Spike count")
    plt.title("Poisson GLM: Spike Counts vs Stimulus (+ fitted mean)")
    plt.tight_layout()
    plt.savefig("poisson_glm_fit.png", dpi=300)
    plt.show()


if __name__ == "__main__":

    run()
