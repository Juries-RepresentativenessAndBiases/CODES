import random
from scipy.optimize import differential_evolution
from scipy.special import comb
from math import log
import numpy as np
from numba import objmode
from scipy.stats import ttest_1samp


# 1. Data sample
# The first coordinate is the mean of votes for acquittal on the first ballot. 
# The second coordinate is the final verdict, being 0 acquittal, 0.5 hung and 1 conviction.
data_sample = (
    [(0, 1)]*43 +
    [(3, 0)]*5 +
    [(3, 0.5)]*10 +
    [(3, 1)]*90 +
    [(6, 0)]*5 +
    [(6, 1)]*5 +
    [(9, 0)]*37 +
    [(9, 0.5)]*3 +
    [(9, 1)]*1 +
    [(12, 0)]*26
)
# Each element corresponds to how many juries had a mean of 0,3,6,9,12 votes for acquittal
data_votes_acquittal=[sum(1 for x in data_sample if x[0]==value) for value in [0,3,6,9,12]]


# 2. Creating the 225-sized samples from the original data (bootstrapped data samples)
n = 100
sample_size = 225
bootstrap_samples = []
for i in range(n):
    sample = random.choices(data_sample, k=sample_size)
    bootstrap_samples.append(sample)


# 3. Defining the MLE and minimum chi^2 methods
# Likelihood Functions
def gamma2(n, i, th, mu):
    return comb(n,i)*(th*(mu**(n-i)*(1-mu)**i) + (1-th)*(1-mu)**(n-i)*mu**i)
def gamma3(n, i, thh, mu1, mu2):
    return comb(n,i)*(thh*(mu1**(n-i)*(1-mu1)**i) + (1-thh)*(1-mu2)**(n-i)*mu2**i)
def neg_log_likelihood2(votes_acquittal, params2):
    th, mu = params2
    log_lik = (
        votes_acquittal[0] * log(gamma2(12, 0, th, mu)) +
        votes_acquittal[1] * sum(log(gamma2(12, i, th, mu)) for i in range(1, 6)) +
        votes_acquittal[2] * log(gamma2(12, 6, th, mu)) +
        votes_acquittal[3] * sum(log(gamma2(12, i, th, mu)) for i in range(7, 12)) +
        votes_acquittal[4] * log(gamma2(12, 12, th, mu))
    )
    return -log_lik
def neg_log_likelihood3(votes_acquittal, params3):
    thh, mu1, mu2 = params3
    log_lik = (
        votes_acquittal[0] * log(gamma3(12, 0, thh, mu1, mu2)) +
        votes_acquittal[1] * sum(log(gamma3(12, i, thh, mu1, mu2)) for i in range(1, 6)) +
        votes_acquittal[2] * log(gamma3(12, 6, thh, mu1, mu2)) +
        votes_acquittal[3] * sum(log(gamma3(12, i, thh, mu1, mu2)) for i in range(7, 12)) +
        votes_acquittal[4] * log(gamma3(12, 12, thh, mu1, mu2))
    )
    return -log_lik

# Minimum chi^2 functions
def chi2_2(votes_acquittal, params2):
    th, mu = params2
    p0 = gamma2(12, 0, th, mu)
    p1 = sum(gamma2(12, i, th, mu) for i in range(1, 6))
    p2 = gamma2(12, 6, th, mu)
    p3 = sum(gamma2(12, i, th, mu) for i in range(7, 12))
    p4 = gamma2(12, 12, th, mu)
    chi2 = ((votes_acquittal[0]/sample_size - p0)**2 / p0 +
            (votes_acquittal[1]/sample_size - p1)**2 / p1 +
            (votes_acquittal[2]/sample_size - p2)**2 / p2 +
            (votes_acquittal[3]/sample_size - p3)**2 / p3 +
            (votes_acquittal[4]/sample_size - p4)**2 / p4)
    return chi2
def chi2_3(votes_acquittal, params3):
    thh, mu1, mu2 = params3
    p03 = gamma3(12, 0, thh, mu1, mu2)
    p13 = sum(gamma3(12, i, thh, mu1, mu2) for i in range(1, 6))
    p23 = gamma3(12, 6, thh, mu1, mu2)
    p33 = sum(gamma3(12, i, thh, mu1, mu2) for i in range(7, 12))
    p43 = gamma3(12, 12, thh, mu1, mu2)
    chi2 = ((votes_acquittal[0]/sample_size - p03)**2 / p03 +
            (votes_acquittal[1]/sample_size - p13)**2 / p13 +
            (votes_acquittal[2]/sample_size - p23)**2 / p23 +
            (votes_acquittal[3]/sample_size - p33)**2 / p33 +
            (votes_acquittal[4]/sample_size - p43)**2 / p43)
    return chi2


# 4. Running the methods
bounds2 = [(0.5, 1.0), (0.5, 1.0)]
bounds3 = [(0.5, 1.0), (0.5, 1.0), (0.5, 1.0)]
mle2_results = []
mle3_results = []
chi2_results = []
chi3_results = []

for sample in bootstrap_samples:
    votes_acquittal = [sum(1 for x in sample if x[0]==value) for value in [0,3,6,9,12]]
    mle2 = differential_evolution(
        lambda params: neg_log_likelihood2(votes_acquittal, params),
        bounds2,
        tol=1e-10,
        strategy='best1bin',
        polish=True,
        updating='deferred',
        seed=42
    )
    mle3 = differential_evolution(
        lambda params: neg_log_likelihood3(votes_acquittal, params),
        bounds3,
        tol=1e-10,
        strategy='best1bin',
        polish=True,
        updating='deferred',
        seed=42
    )
    chi2 = differential_evolution(
        lambda params: chi2_2(votes_acquittal, params),
        bounds2,
        tol=1e-10,
        strategy='best1bin',
        polish=True,
        updating='deferred',
        seed=42
    )
    chi3 = differential_evolution(
        lambda params: chi2_3(votes_acquittal, params),
        bounds3,
        tol=1e-10,
        strategy='best1bin',
        polish=True,
        updating='deferred',
        seed=42
    )
    mle2_results.append(mle2.x)
    mle3_results.append(mle3.x)
    chi2_results.append(chi2.x)
    chi3_results.append(chi3.x)

# differential_evolution() finds minimums. It works well because it explores all the range inside the boundaries
# and does not get stucked in local minimums. It works with a population-based evolutionary approach, which
# is nothing more than starting with a population of candidates, instead of one, and creating a new set of
# new candidates by subtracting older ones, and from all these ones, selecting to "survive" the ones that
# give a lower value of our function.
 
# The parameter "polish=True" helps with the accuracy of the final solution,
# "updating='deferred'" makes the method more numerically stable because between steps of the evolution
# of the population, it makes the method wait until all members have been evaluated (can make the method
# slower, but in our case we are insterested in accuracy, not computational efficiency),
# "strategy='best1bin'" takes the best solution in the moment, mutates it a bit with other elements of the
# population and compares these candidates, so it gives more accuracy around the possible best solution 


# 5. Difference in parameters between the MLE and minimum chi^2 methods + saving data
theta_mle2 = [x[0] for x in mle2_results]
mu_mle2 = [x[1] for x in mle2_results]
theta_mle3 = [x[0] for x in mle3_results]
mu1_mle3 = [x[1] for x in mle3_results]
mu2_mle3 = [x[2] for x in mle3_results]

theta_chi2 = [x[0] for x in chi2_results]
mu_chi2 = [x[1] for x in chi2_results]
theta_chi3 = [x[0] for x in chi3_results]
mu1_chi3 = [x[1] for x in chi3_results]
mu2_chi3 = [x[2] for x in chi3_results]

theta_mle2 = np.array(theta_mle2)
mu_mle2 = np.array(mu_mle2)
theta_mle3 = np.array(theta_mle3)
mu1_mle3 = np.array(mu1_mle3)
mu2_mle3 = np.array(mu2_mle3)
theta_chi2 = np.array(theta_chi2)
mu_chi2 = np.array(mu_chi2)
theta_chi3 = np.array(theta_chi3)
mu1_chi3 = np.array(mu1_chi3)
mu2_chi3 = np.array(mu2_chi3)

with objmode():
    np.savez(f"{n}-sized_bootstrap_estimates_TFGM", 
             n = n,
             theta_mle2 = theta_mle2,
             mu_mle2 = mu_mle2,
             theta_mle3 = theta_mle3,
             mu1_mle3 = mu1_mle3,
             mu2_mle3 = mu2_mle3,
             theta_chi2 = theta_chi2,
             mu_chi2 = mu_chi2,
             theta_chi3 = theta_chi3,
             mu1_chi3 = mu1_chi3,
             mu2_chi3 = mu2_chi3
             )

theta_diff2 = theta_mle2 - theta_chi2
mu_diff2 = mu_mle2 - mu_chi2
theta_diff3 = theta_mle3 - theta_chi3
mu1_diff3 = mu1_mle3 - mu1_chi3
mu2_diff3 = mu2_mle3 - mu2_chi3


# 6. Running a t-test to check the difference in parameters between the MLE and minimum chi^2 methods
# (H_0: mean = 0)
print("== Model 2: (theta, mu) ==")
t_theta2 = ttest_1samp(theta_diff2, 0)
t_mu2 = ttest_1samp(mu_diff2, 0)
print(f"Theta t-test: t={t_theta2.statistic:.4f}, p={t_theta2.pvalue:.2e}")
print(f"Mu    t-test: t={t_mu2.statistic:.4f}, p={t_mu2.pvalue:.2e}")

print("\n== Model 3: (theta, mu1, mu2) ==")
t_theta3 = ttest_1samp(theta_diff3, 0)
t_mu13 = ttest_1samp(mu1_diff3, 0)
t_mu23 = ttest_1samp(mu2_diff3, 0)
print(f"Theta t-test: t={t_theta3.statistic:.4f}, p={t_theta3.pvalue:.2e}")
print(f"Mu1   t-test: t={t_mu13.statistic:.4f}, p={t_mu13.pvalue:.2e}")
print(f"Mu2   t-test: t={t_mu23.statistic:.4f}, p={t_mu23.pvalue:.2e}")