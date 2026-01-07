"""
Population Coding with Tuning Curves

To simulate a population of neurons with Gaussian tuning curves and
    visualise how the population represents a stimulus.

Source:
    Neuromatch Academy - Neural Coding / Population Coding
"""

import numpy as np
import matplotlib.pyplot as plt


def gaussian_tuning(stimulus, pref, sigma, r_max=50.0, baseline=5.0):
    """
    Gaussian tuning curve:
    response = baseline + r_max * exp(-(stim - pref)^2 / (2*sigma^2))
    stimulus, pref, sigma in same units (e.g., degrees).
    returns firing rate in Hz.
    """
    return baseline + r_max * np.exp(-((stimulus - pref) ** 2) / (2 * sigma ** 2))


def run():
    # 1) Define stimulus space (e.g., orientation in degrees)
    stim_space = np.linspace(-90, 90, 361)

    # 2) Create a population of neurons with different preferred stimuli
    n_neurons = 12
    prefs = np.linspace(-75, 75, n_neurons)   # preferred orientations
    sigma = 20.0                               # tuning width (degrees)

    # 3) Plot tuning curves
    plt.figure()
    for pref in prefs:
        rates = gaussian_tuning(stim_space, pref=pref, sigma=sigma)
        plt.plot(stim_space, rates)
    plt.xlabel("Stimulus (degrees)")
    plt.ylabel("Firing rate (Hz)")
    plt.title("Neural Population Tuning Curves")
    plt.tight_layout()
    plt.savefig("population_tuning_curves.png", dpi=300)
    plt.show()

    # 4) Pick a stimulus and show population response pattern
    stimulus = 20.0
    pop_response = gaussian_tuning(stimulus, pref=prefs, sigma=sigma)

    plt.figure()
    plt.stem(prefs, pop_response, basefmt=" ")
    plt.xlabel("Preferred stimulus (degrees)")
    plt.ylabel("Firing rate (Hz)")
    plt.title(f"Population Response Pattern (Stimulus = {stimulus:.1f}°)")
    plt.tight_layout()
    plt.savefig("population_response_pattern.png", dpi=300)
    plt.show()

    # 5) (Optional but nice) Simple population-vector estimate of stimulus
    # This is a basic decoder: weighted average of preferences by firing rate.
    stim_est = np.sum(prefs * pop_response) / np.sum(pop_response)
    print(f"True stimulus: {stimulus:.1f}°, Population-vector estimate: {stim_est:.1f}°")


if __name__ == "__main__":
    run()
