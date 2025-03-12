import matplotlib.pyplot as plt
import pandas as pd

# Data
data = {
    "tinit": [100, 100, 100, 100, 200, 200, 200, 200, 300, 300, 300, 300, 400, 400, 400, 400, 500, 500, 500, 500],
    "lambda": [0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5],
    "gamma": [0, 0.01, 0, 0.01, 0, 0.01, 0, 0.01, 0, 0.01, 0, 0.01, 0, 0.01, 0, 0.01, 0, 0.01, 0, 0.01],
    "picp_mean": [95.32072449, 94.93078613, 94.96977997, 94.56034088, 95.22768402, 94.98905945, 95.04872131, 94.81010437,
                  95.49604797, 95.2322998, 95.49604797, 95.21201324, 96.68668365, 95.34065247, 96.37606049, 95.44419098,
                  96.405159, 95.30556488, 96.405159, 95.43243408],
    "eff_mean": [90.95895467, 79.64195666, 72.69721342, 61.69217514, 90.44046378, 81.14211229, 74.48378568, 64.20952304,
                 90.83644748, 80.6145359, 82.62722643, 70.71823553, 91.40238233, 78.13295596, 82.70154124, 70.63311752,
                 89.89629757, 78.42212547, 73.65463411, 63.36856208]
}

df = pd.DataFrame(data)

# Unique lambda-gamma pairs for legend
lambda_gamma_pairs = df[['lambda', 'gamma']].drop_duplicates().values
print(lambda_gamma_pairs)

# Define distinct colors for the four (lambda, gamma) combinations
colors = ['blue', 'royalblue', 'red', 'lightcoral']
labels = ['MultiDimSPCI($\gamma=0$)',  'MultiDimSPCI($\gamma=0.01$)', 'STACI ($\gamma=0$)','STACI($\gamma=0.01$)']
tick_fontsize = 13
title_fontsize = 15

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left plot: picp_mean

for (idx, (lmb, gma)) in enumerate(lambda_gamma_pairs):
    subset = df[(df['lambda'] == lmb) & (df['gamma'] == gma)]
    axes[0].plot(subset["tinit"], subset["picp_mean"], marker='o', color=colors[idx], label=labels[idx])


axes[0].set_xlabel("$$n$$", fontsize=tick_fontsize)
axes[0].set_ylabel("Coverage (%)", fontsize=tick_fontsize)
axes[0].tick_params(labelsize=tick_fontsize)
axes[0].set_title("Coverage Rate of Offline Methods", fontsize=title_fontsize)
axes[0].legend()

# Right plot: eff_mean
for (idx, (lmb, gma)) in enumerate(lambda_gamma_pairs):
    subset = df[(df['lambda'] == lmb) & (df['gamma'] == gma)]
    axes[1].plot(subset["tinit"], subset["eff_mean"], marker='o', color=colors[idx], label=labels[idx])

axes[1].set_xlabel("$$n$$", fontsize=tick_fontsize)
axes[1].set_ylabel("Efficiency", fontsize=tick_fontsize)
axes[1].tick_params(labelsize=tick_fontsize)
axes[1].set_title("Efficiency of Offline Methods", fontsize=title_fontsize)
axes[0].set_xticks([100, 200, 300, 400, 500])
axes[1].set_xticks([100, 200, 300, 400, 500])
# axes[1].legend()

axes[1].set_ylim(axes[1].get_ylim()[::-1])

axis_width = 2.5
for ax in axes:
    ax.spines['bottom'].set_linewidth(axis_width)  # Bold X-axis
    ax.spines['left'].set_linewidth(axis_width)  # Bold Y-axis
    ax.spines['top'].set_linewidth(axis_width)  # Bold Top border (optional)
    ax.spines['right'].set_linewidth(axis_width)  # Bold Right border (optional)



plt.tight_layout()
plt.savefig('logs/ablation.png')
plt.show()

