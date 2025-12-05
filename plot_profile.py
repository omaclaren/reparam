import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.stats import chi2
import csv

print("Loading data...")

# Load grid data
beta1 = []
K1 = []
lnlike = []

with open('profile_data.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        beta1.append(float(row[0]))
        K1.append(float(row[1]))
        lnlike.append(float(row[2]))

beta1 = np.array(beta1)
K1 = np.array(K1)
lnlike = np.array(lnlike)

# Normalize likelihood
max_lnlike = np.max(lnlike)
lnlike_norm = lnlike - max_lnlike
like = np.exp(lnlike_norm)

# Load metadata
mle_pt = None
true_pt = None

with open('metadata.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader) # skip header
    for row in reader:
        if row[0] == 'MLE':
            mle_pt = (float(row[1]), float(row[2]))
        elif row[0] == 'TRUE':
            true_pt = (float(row[1]), float(row[2]))

print("Plotting...")

# Create plot
plt.figure(figsize=(10, 8))

# Create grid for contour plot
# Since data might not be perfectly structured if we just take lists,
# using triangulation is safer for arbitrary points, but tricontourf works well.
cntr = plt.tricontourf(beta1, K1, like, levels=30, cmap='BuPu')

# Add colorbar
cbar = plt.colorbar(cntr)
cbar.set_label('Profile Likelihood', fontsize=14)

# Add 95% confidence contour
df = 2
lstar = np.exp(-chi2.ppf(0.95, df) / 2)
plt.tricontour(beta1, K1, like, levels=[lstar], colors='k', linewidths=1.5)

# Add points
if mle_pt:
    plt.plot(mle_pt[0], mle_pt[1], 'o', color='silver', markeredgecolor='k', markersize=10, label='MLE')

if true_pt:
    plt.plot(true_pt[0], true_pt[1], '*', color='goldenrod', markeredgecolor='k', markersize=15, label='True')

plt.xlabel(r'$\beta_1$', fontsize=16)
plt.ylabel(r'$K_1$', fontsize=16)
plt.title(r'Repressilator 2D Profile: $(\beta_1, K_1)$', fontsize=18)
plt.legend(loc='lower right', fontsize=12)

# Save
filename = 'repressilator_2D_profile_python.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"Saved to {filename}")
