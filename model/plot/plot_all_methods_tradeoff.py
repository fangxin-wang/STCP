import matplotlib.pyplot as plt
import numpy as np

# Data from the table
methods = [
    "Sphere", "Sphere-ACI (γ=0.01)", "Square", "GT",
    "MultiDimSPCI", "STACI(γ=0)", "STACI(γ=0.01)"
]
cov_l = [93,94,95,96,97]

# Coverage values
coverage_theta_00 = [95.223, 95.000, 97.315, 95.193, 93.376, 94.532, 95.000]
coverage_theta_07 = [95.532, 95.021, 97.247, 95.445, 93.890, 94.850, 94.992]

# Inefficiency values (convert to negative for plotting)
inefficiency_theta_00 = [-3.817, -3.819, -4.350, -2.433, -2.260, -2.342, -2.379]
inefficiency_theta_07 = [-3.877, -3.850, -4.381, -2.443, -2.275, -2.352, -2.374]

# Approximate coverage covariance by using standard deviation of coverage as a proxy for size
covariance_theta_00 = [0.006, 0.002, 0.021, 0.007, 0.076, 0.039, 0.001]
covariance_theta_07 = [0.068, 0.001, 0.020, 0.040, 0.067, 0.024, 0.003]

# Normalize point sizes for better visualization
point_sizes_00 = [cov * 10000 + 20 for cov in covariance_theta_00]
point_sizes_07 = [cov * 10000 + 20 for cov in covariance_theta_07]

# Define font size parameters
title_fontsize = 18
label_fontsize = 16
annotate_fontsize = 14

axis_width = 2.5

# Adjust label position to avoid overlap
offset = [(-0.05,0.1), (-1.5,-0.28), (-0.8,0.1), (0.2,0),
          (-0.35,-0.29), (0.05,0.05), (-0.15,-0.32)]  # Offset factor for annotation placement

# Define distinct colors for each method
# colors = [
#     'blue', 'deepskyblue', 'purple', 'tomato', 'orange', 'green', 'darkgreen'
# ]
colors = ['grey', 'grey', 'grey', 'grey', 'grey', 'tomato', 'red']
# Create scatter plots with adjustable font sizes and colors
fig, axes = plt.subplots(1, 2, figsize=(8.5,5))

# Plot for Theta = (0,0)
for i, method in enumerate(methods):
    if i ==3:
        axes[0].plot(coverage_theta_00[i], inefficiency_theta_00[i], marker = "*",  markersize = 12, color = 'green')
    else:
        axes[0].scatter(
            coverage_theta_00[i], inefficiency_theta_00[i],
            s=point_sizes_00[i], color=colors[i], alpha=0.5, edgecolors='black'
        )
    axes[0].annotate(
        method,
        (coverage_theta_00[i] + offset[i][0], inefficiency_theta_00[i] + offset[i][1]),
        fontsize=annotate_fontsize,
        ha='left',
        va='bottom',
        color=colors[i]
    )
axes[0].set_title(r'$\Theta = (0,0)$', fontsize=title_fontsize)
axes[0].set_xlabel('Coverage (%)', fontsize=label_fontsize)
axes[0].set_ylabel('Efficiency', fontsize=label_fontsize)
axes[0].tick_params(axis='both', labelsize=label_fontsize)


# Plot for Theta = (0.7,0.3)
for i, method in enumerate(methods):
    print(i, method)
    if i ==3:
        axes[1].plot( coverage_theta_07[i], inefficiency_theta_07[i], marker = "*", markersize = 12, color = 'green')
    else:
        axes[1].scatter(
            coverage_theta_07[i], inefficiency_theta_07[i],
            s=point_sizes_07[i], color=colors[i], alpha=0.5, edgecolors='black'
        )
    axes[1].annotate(
        method,
        (coverage_theta_07[i] + offset[i][0], inefficiency_theta_07[i] + offset[i][1]),
        fontsize=annotate_fontsize,
        ha='left',
        va='bottom',
        color=colors[i]
    )
axes[1].set_title(r'$\Theta = (0.7,0.3)$', fontsize=title_fontsize)
axes[1].set_xlabel('Coverage (%)', fontsize=label_fontsize)
axes[1].tick_params(axis='both', labelsize=label_fontsize, left=False, labelleft=False)  # Hide y-axis ticks for right plot
# axes[1].tick_params(axis='both', labelsize=label_fontsize)

# axes[0].axvline(x=95, color='grey', linestyle='dotted', linewidth=2)
# axes[1].axvline(x=95, color='grey', linestyle='dotted', linewidth=2)
#
# axes[0].set_xticks(cov_l)
# axes[1].set_xticks(cov_l)
# axes[0].set_xlim( 93,97.5)
# axes[1].set_xlim( 93,97.5)
#
# axes[0].set_ylim( -4.5, -2)
# axes[1].set_ylim( -4.5, -2)
# axes[0].set_yticklabels()

for ax in axes:
    ax.axvline(x=95, color='grey', linestyle='dotted', linewidth=2)
    ax.set_xticks(cov_l)
    ax.set_xlim(93, 97.5)
    ax.set_ylim(-4.5, -2)
    ax.set_yticks([-4.5,-4.0,-3.5,-3.0,-2.5,-2.0])
    ax.set_yticklabels([4.5,4.0,3.5,3.0,2.5,2.0])


# for ax in axes:
#     # ax.annotate('better', (96.7, -2.65), fontsize = 15, color = 'red')
#     ax.arrow(97,-2.4,0.2,0.1, head_width = 0.1, width = 0.05, color = 'red')

for ax in axes:
    ax.spines['bottom'].set_linewidth(axis_width)  # Bold X-axis
    ax.spines['left'].set_linewidth(axis_width)  # Bold Y-axis
    ax.spines['top'].set_linewidth(axis_width)  # Bold Top border (optional)
    ax.spines['right'].set_linewidth(axis_width)  # Bold Right border (optional)

legend_variances = [0.001, 0.01, 0.05]  # Example variance values for legend
legend_sizes = [v * 10000 + 20 for v in legend_variances]  # Scale for visualization
for var, size in zip(legend_variances, legend_sizes):
    axes[0].scatter([], [], s=size, color="gray", alpha=0.5, label=f"{var}")
axes[0].legend(title="Coverage Var.", fontsize=12, bbox_to_anchor=(0, 0.7), loc="upper left", title_fontsize=12)

# Show the plots with updated font sizes, colors, and improved label positioning
plt.tight_layout()
plt.savefig('logs/comparison.png')
# plt.show()



# Re-define data after execution state reset
methods_pems = [
    "Sphere", "Sphere-ACI (γ=0.01)", "Square",
    "MultiDimSPCI", "STACI(γ=0)", "STACI(γ=0.01)"
]

coverage_pems = [96.130, 95.179, 97.527, 91.246, 95.940, 95.432]
inefficiency_pems = [-126.239, -113.495, -169.294, -75.050, -80.070, -69.968]  # Negative inefficiency for plotting

covariance_pems = [12.975, 21.945, 20.210, 7.355, 13.230, 9.747]  # Using std dev as a proxy for variance
point_sizes_pems = [cov * 25 for cov in covariance_pems]  # Scale for visualization


# Define distinct colors for each method
colors_pems = ['grey', 'grey', 'grey', 'grey', 'tomato', 'red']
offset_pems = [(-0.05,3), (-1,4), (-0.8,4),
          (0.2,-0.6), (0.2,-2), (0.2,-2.5)]

# Create scatter plot for PeMS dataset
fig, ax = plt.subplots(figsize=(6,4))

for i, method in enumerate(methods_pems):
    ax.scatter(
        coverage_pems[i], inefficiency_pems[i],
        s=point_sizes_pems[i], color=colors_pems[i], alpha=0.5, edgecolors='black'
    )
    ax.annotate(
        method,
        (coverage_pems[i] + offset_pems[i][0], inefficiency_pems[i] + offset_pems[i][1]),
        fontsize=annotate_fontsize,
        ha='left',
        va='bottom',
        color=colors_pems[i]
    )

# Add elements to the plot
# ax.set_title(r'Comparison of Methods for PeMS Dataset', fontsize=title_fontsize)
ax.set_xlabel('Coverage (%)', fontsize=label_fontsize)
ax.set_ylabel('Efficiency', fontsize=label_fontsize)
ax.tick_params(axis='both', labelsize=label_fontsize)

# Add coverage threshold line at 94.5%
ax.axvline(x=95, color='grey', linestyle='dotted', linewidth=2)

ax.spines['bottom'].set_linewidth(axis_width)  # Bold X-axis
ax.spines['left'].set_linewidth(axis_width)  # Bold Y-axis
ax.spines['top'].set_linewidth(axis_width)  # Bold Top border (optional)
ax.spines['right'].set_linewidth(axis_width)  # Bold Right border (optional)

# ax.arrow(91, -170, 0.5, 5, head_width=0.4, width=0.1, color='red')
legend_variances = [5, 10, 20]  # Example variance values for legend
legend_sizes = [v * 25 for v in legend_variances]  # Scale for visualization
for var, size in zip(legend_variances, legend_sizes):
    ax.scatter([], [], s=size, color="gray", alpha=0.5, label=f"{var}")
ax.legend(title="Efficiency Var.", fontsize=12, bbox_to_anchor=(0, 0), loc="lower left", title_fontsize=12)

ax.set_ylim(-180,-60)
tickers = np.arange(60, 200, step=20)
ax.set_yticks(-tickers)
ax.set_yticklabels(tickers)
# Show the plot
plt.tight_layout()

plt.savefig('logs/comparison_pems.png')
plt.show()


