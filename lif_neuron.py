"""
Leaky Integrate-and-Fire (LIF) Neuron Model

To simulate a single leaky integrate-and-fire neuron and
    examine how its firing rate depends on input current.

Source: Neuromatch Academy Intro to Modelling
"""

import numpy as np
import matplotlib.pyplot as plt

# Neuron parameters
tau_m = 20e-3       # membrane time constant (s)
R_m = 10e6          # membrane resistance (ohms)
V_rest = -65e-3     # resting potential (V)
V_th = -50e-3       # spike threshold (V)
V_reset = -65e-3    # reset potential (V)

dt = 1e-4           # time step (s)
T = 1.0             # total simulation time (s)
time = np.arange(0, T, dt)

#simulation function
def simulate_lif(I):
    V = np.zeros_like(time)
    V[0] = V_rest
    spikes = []

    for t in range(1, len(time)):
        dV = (-(V[t-1] - V_rest) + R_m * I) / tau_m
        V[t] = V[t-1] + dV * dt

        if V[t] >= V_th:
            V[t] = V_reset
            spikes.append(time[t])

    return V, spikes

#run one example simulation
I_example = 2e-9  # input current (A)
V, spikes = simulate_lif(I_example)

plt.figure()
plt.plot(time, V * 1e3)
plt.axhline(V_th * 1e3, linestyle="--", label="Threshold")
plt.xlabel("Time (s)")
plt.ylabel("Membrane potential (mV)")
plt.title("LIF Neuron Membrane Potential")
plt.legend()
plt.tight_layout()
plt.savefig("lif_membrane_trace.png", dpi=300)
plt.show()

# Firing rate vs input current
currents = np.linspace(0.5e-9, 3e-9, 10)
firing_rates = []

for I in currents:
    _, spikes = simulate_lif(I)
    firing_rates.append(len(spikes) / T)

plt.figure()
plt.plot(currents * 1e9, firing_rates, marker="o")
plt.xlabel("Input current (nA)")
plt.ylabel("Firing rate (Hz)")
plt.title("Firing Rate vs Input Current")
plt.tight_layout()
plt.savefig("lif_firing_rate.png", dpi=300)
plt.show()
