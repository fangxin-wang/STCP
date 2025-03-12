import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Filter only gamma == 0
gamma = 0 #  0.01 #0
LEFT = False
tinit = 300
seed =  73 # 73   #30
axis_width = 2.5

# data = pd.read_csv(f'syn{seed}_res_all.csv')
data = pd.read_csv(f'grained_lambda_syn_seed{seed}.csv')
df = pd.DataFrame(data)

df_filtered = df[df["gamma"] == gamma][df['tinit']==tinit].sort_values(by='lambda', ascending=True)

df_filtered = df_filtered.copy()  # Avoid SettingWithCopyWarning
df_filtered["lambda"] = pd.to_numeric(df_filtered["lambda"], errors='coerce')
df_filtered["picp_mean"] = pd.to_numeric(df_filtered["picp_mean"], errors='coerce')
df_filtered["eff_mean"] = pd.to_numeric(-df_filtered["eff_mean"], errors='coerce')
print(df_filtered.max(),df_filtered.min())
df_filtered["eff_std"] = pd.to_numeric(df_filtered["eff_std"], errors='coerce')

# Convert all necessary columns to numpy arrays to ensure compatibility with matplotlib
df_filtered["lambda"] = df_filtered["lambda"].astype(np.float64)
df_filtered["eff_mean"] = df_filtered["eff_mean"].astype(np.float64)
df_filtered["eff_std"] = df_filtered["eff_std"].astype(np.float64)
df_filtered["picp_mean"] = df_filtered["picp_mean"].astype(np.float64)

# Create figure and axis with dual y-axis
fig, ax1 = plt.subplots(figsize=(4,4))

# Secondary y-axis
ax2 = ax1.twinx()

tick_fontsize = 13
legend_fontsize =10
linewidth = 2
axis_label_fontsize = 15
title_fontsize =15
markersize = 1

# Define colors and linestyles
# colors = {"sphere": "blue", "square": "green", "ellip": "red", "GT": "dimgrey"}
labels = {"sphere": "Sphere", "square": "Square", "ellip": "STACI", "GT": "GT"}
# linestyles = {"sphere": "--", "square": "--", "ellip": "-"}

# Plot data
# for cov in df_filtered["cov_type"].unique():
for cov in ["GT","ellip"]:
    subset = df_filtered[df_filtered["cov_type"] == cov]
    print(subset)
    if cov in ["sphere", "square","GT"]:
        # Horizontal lines for sphere and square
        # ax1.axhline(y=subset["picp_mean"].iloc[0], color=colors[cov], linestyle='dashed',linewidth= linewidth, label= labels[cov] )
        # ax2.axhline(y=subset["eff_mean"].iloc[0], color=colors[cov], linestyle='solid' ,linewidth= linewidth, alpha=0.7, label= labels[cov])

        ax1.axhline(y=subset["picp_mean"].iloc[0], color= 'red', linestyle='dashed', linewidth=linewidth,
                    label=labels[cov])
        ax2.axhline(y=subset["eff_mean"].iloc[0], color= 'green', linestyle='dashed', linewidth=linewidth, alpha=0.7,
                    label=labels[cov])
    else:
        # Curve for ellip
        # ax1.plot(subset["lambda"], subset["picp_mean"], linestyle= 'dashed', color=colors[cov],linewidth= linewidth, marker='o', label= labels[cov])
        # ax2.plot(subset["lambda"], subset["eff_mean"], linestyle='solid', color=colors[cov],linewidth= linewidth, marker='s', alpha=0.7, label= labels[cov])
        # ax2.fill_between(subset["lambda"].to_numpy(),
        #                  (subset["eff_mean"] - subset["eff_std"]).to_numpy(),
        #                  (subset["eff_mean"] + subset["eff_std"]).to_numpy(),
        #                  color=colors[cov], alpha=0.1)
        ax1.plot(subset["lambda"], subset["picp_mean"], linestyle='solid', color= 'red', linewidth=linewidth,
                 marker='o', markersize = markersize, label=labels[cov])
        ax2.plot(subset["lambda"], subset["eff_mean"], linestyle='solid', color= 'green', linewidth=linewidth,
                 marker='s', markersize = markersize, alpha=0.7, label=labels[cov])
        ax2.fill_between(subset["lambda"].to_numpy(),
                         (subset["eff_mean"] - subset["eff_std"]).to_numpy(),
                         (subset["eff_mean"] + subset["eff_std"]).to_numpy(),
                         color= 'green', alpha=0.05)

ax1.set_xlabel("$\lambda$", fontsize=axis_label_fontsize)


if LEFT:
    ax2.get_yaxis().set_visible(False)
    ax1.set_ylabel("Coverage(%)", fontsize=axis_label_fontsize)
else:
    ax1.get_yaxis().set_visible(False)
    ax2.set_ylabel("Efficiency", fontsize=axis_label_fontsize)
    ax1.legend(title='Coverage', loc="lower left")
    ax2.legend(title='Efficiency', loc="lower right")

plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
ax1.tick_params(axis='both', labelsize=tick_fontsize)
ax2.tick_params(axis='both', labelsize=tick_fontsize)

for spine in ax1.spines.values():
    spine.set_linewidth(2)

left_spine1, right_spine1, left_spine2, right_spine2 = ax1.spines['left'], ax1.spines['right'], ax2.spines['left'], ax2.spines['right']
left_spine1.set_color('red')
left_spine2.set_color('red')
right_spine1.set_color('green')
right_spine2.set_color('green')

ax1.set_ylim(93, 96)
ax2.set_ylim(-3, -2)
tickers = [-3,-2.8,-2.6,-2.4,-2.2,-2]
ax2.set_yticks(tickers)
ax2.set_yticklabels([-i for i in tickers])

# ax1.legend(title = 'Coverage', bbox_to_anchor=(0, 0.9), loc="upper left", fontsize=legend_fontsize)
# ax2.legend(title = 'Efficiency',bbox_to_anchor=(1, 0.9),loc="upper right", fontsize=legend_fontsize)

ax = plt.gca()
ax.spines['bottom'].set_linewidth(axis_width)  # Bold X-axis
ax.spines['left'].set_linewidth(axis_width)  # Bold Y-axis
ax.spines['top'].set_linewidth(axis_width)  # Bold Top border (optional)
ax.spines['right'].set_linewidth(axis_width)  # Bold Right border (optional)

plt.title(r'$\Theta = (0.7, 0.3)$'.format(gamma) )

# plt.title(r'$\Theta = (0.7, 0.3), \gamma = {}$'.format(gamma), fontsize=title_fontsize)

plt.tight_layout()
plt.savefig('logs/syn{}_gamma_{}_tinit_{}_tradeoff.png'.format(seed,gamma,tinit))

# Show plot
plt.show()