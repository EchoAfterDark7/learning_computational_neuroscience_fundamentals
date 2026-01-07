"""
Title: Linear Decoding of Neural Population Activity

Goal:
    To simulate population responses for two stimulus classes and train a
    linear classifier (logistic regression) to decode class from neural activity.

Source:
    Neuromatch Academy – Machine Learning / Decoding
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA


def gaussian_tuning(stimulus, pref, sigma, r_max=50.0, baseline=5.0):
    return baseline + r_max * np.exp(-((stimulus - pref) ** 2) / (2 * sigma ** 2))


def simulate_trial(stimulus, prefs, sigma, noise_std=10.0):
    """
    Returns one population response vector (Hz) for a given stimulus.
    Adds Gaussian noise to firing rates.
    """
    rates = gaussian_tuning(stimulus, pref=prefs, sigma=sigma)
    noisy = rates + np.random.normal(0.0, noise_std, size=rates.shape)
    return np.clip(noisy, 0.0, None)  # no negative firing rates


def run():
    rng = np.random.default_rng(0)

    # 1) Build a population
    n_neurons = 30
    prefs = np.linspace(-75, 75, n_neurons)
    sigma = 30.0

    # 2) Create two classes of stimuli
    # Class 0 centered around -20°, Class 1 centered around +20°
    n_trials_per_class = 300
    class0_stim = rng.normal(loc=-20.0, scale=15.0, size=n_trials_per_class)
    class1_stim = rng.normal(loc=+20.0, scale=15.0, size=n_trials_per_class)

    # 3) Simulate population activity matrix X and labels y
    X0 = np.array([simulate_trial(s, prefs, sigma) for s in class0_stim])
    X1 = np.array([simulate_trial(s, prefs, sigma) for s in class1_stim])
    X = np.vstack([X0, X1])
    y = np.array([0] * n_trials_per_class + [1] * n_trials_per_class)

    # 4) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )

    # 5) Standardise features (important for linear models)
    scaler = StandardScaler()
    X_train_z = scaler.fit_transform(X_train)
    X_test_z = scaler.transform(X_test)

    # 6) Train a linear decoder
    clf = LogisticRegression(max_iter=1000, random_state=0)
    clf.fit(X_train_z, y_train)

    # 7) Evaluate
    y_pred = clf.predict(X_test_z)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.3f}")

    # 8) Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Class 0", "Class 1"])
    disp.plot()
    plt.title("Confusion Matrix: Linear Decoding")
    plt.tight_layout()
    plt.savefig("linear_decoding_confusion_matrix.png", dpi=300)
    plt.show()

    # 9) Optional: PCA plot for visual intuition (not required, but looks great)
    pca = PCA(n_components=2, random_state=0)
    X_2d = pca.fit_transform(scaler.fit_transform(X))

    plt.figure()
    plt.scatter(X_2d[y == 0, 0], X_2d[y == 0, 1], alpha=0.6, label="Class 0")
    plt.scatter(X_2d[y == 1, 0], X_2d[y == 1, 1], alpha=0.6, label="Class 1")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.title("PCA of Population Activity (Two Classes)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("linear_decoding_pca.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    run()
