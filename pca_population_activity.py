"""
PCA of Neural Population Activity

To apply Principal Component Analysis (PCA) to high-dimensional neural population responses and visualise dominant patterns of variance.

Source:
    Neuromatch Academy - Dimensionality Reduction
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def gaussian_tuning(stimulus, pref, sigma, r_max=40.0, baseline=5.0):
    return baseline + r_max * np.exp(-((stimulus - pref) ** 2) / (2 * sigma ** 2))


def run():
    rng = np.random.default_rng(0)

    
    # 1 Population parameters
    
    n_neurons = 40
    prefs = np.linspace(-90, 90, n_neurons)
    sigma = 35.0

    
    # 2 Generate stimuli
    
    n_trials = 800
    stimuli = rng.uniform(-60, 60, size=n_trials)

    
    # 3 Simulate population activity
    
    X = []
    for stim in stimuli:
        rates = gaussian_tuning(stim, pref=prefs, sigma=sigma)
        noisy = rates + rng.normal(0, 6.0, size=rates.shape)
        X.append(np.clip(noisy, 0.0, None))

    X = np.array(X)  # shape: trials x neurons

    
    # 4 Standardise + PCA
    
    X_z = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_z)

    print("Explained variance ratio:", pca.explained_variance_ratio_)

    
    # 5 Visualisation
    
    plt.figure()
    sc = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=stimuli,
        cmap="viridis",
        alpha=0.7
    )
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.title("PCA of Neural Population Activity")
    plt.colorbar(sc, label="Stimulus value")
    plt.tight_layout()
    plt.savefig("pca_population_activity.png", dpi=300)
    plt.show()

    
    # 6 Variance explained plot
    
    pca_full = PCA()
    pca_full.fit(X_z)

    plt.figure()
    plt.plot(np.cumsum(pca_full.explained_variance_ratio_), marker="o")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA Cumulative Explained Variance")
    plt.tight_layout()
    plt.savefig("pca_variance_explained.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    run()

