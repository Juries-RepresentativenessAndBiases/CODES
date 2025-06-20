import numpy as np
import matplotlib.pyplot as plt


# 1. Defining some variables
mu = 0.7494
num_trials = 10000  # number of simulated trials per theta
theta_vals = np.linspace(0, 1, 50)
jury_configs = [(12, 0), (12, 1), (12, 2), (6, 1), (6, 0)]


# 2. Function that simulates the votes of juries and computes Gamma_{n,i} empirically
def simulate_Gamma(n, i, theta, mu, num_trials):
    guilt_status = np.random.rand(num_trials) < theta    # Assigning the guilt status (False or True) of each accused by the prob. theta
    prob_conviction = np.where(guilt_status, mu, 1 - mu) # Assigning mu or 1-mu if guilt_status[i] is True or False
    jury_votes = np.random.rand(num_trials, n) < prob_conviction[:, None] # Assigning n votes for conviction per trial (True == vote for conviction) by the correct probability
    votes_for_conviction = np.sum(jury_votes, axis=1)    # Sum of all the votes for conviction per trial
    m = n - i                                            # Size of the required majority to convict
    convictions = votes_for_conviction >= m              # Assigning the status of conviciton per trial (True == conviction)
    Gamma_estimate = np.mean(convictions)
    return Gamma_estimate


# 3. Plotting
fig, ax = plt.subplots(figsize=(6, 5))
for n, i in jury_configs:
    Gamma_vals = [simulate_Gamma(n, i, theta, mu, num_trials) for theta in theta_vals]
    label = f'{n - i} out of {n} jury'
    plt.plot(theta_vals, Gamma_vals, label=label)


# 4. Styling of borders, legend, and axes + saving and showing the plot
def set_gray_ticks_and_borders(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('gray')
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', length=6, width=1, direction='out', color='gray', top=True, right=True)
    ax.tick_params(axis='both', which='minor', length=3, width=1, direction='out', color='gray', top=True, right=True)
    ax.tick_params(labelcolor='black')

plt.xlabel(r'$\theta$', fontsize=14)
plt.ylabel(r'$\Gamma_{n,i}$', fontsize=14)
plt.legend(fontsize=11)
ax.grid(False)
set_gray_ticks_and_borders(ax)
plt.tight_layout()
#plt.savefig("Gamma_vs_theta_simu.png", dpi=300)
plt.show()