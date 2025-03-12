import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# ---------------------------------------------------
# 1. Example data (placeholder)
# ---------------------------------------------------
# Suppose you have the same set of lambda values
# for each method and each metric.
# Here, we just make up some example values.

# seed = 73
gamma = 0 #0 #0.01
# data = pd.read_csv("results_syn_tailup_gen_seed{}.csv".format(seed) )

data = pd.read_csv( 'grained_lambda_PEMS03.csv')
data = data[ data['gamma'] == gamma]
print(data)

ellip_data_fixed = data[ (data['cov_type'] == 'ellip') ].sort_values(by='lambda', ascending=True)
# ellip_data_fixed['eff_mean'] = -ellip_data_fixed['eff_mean']
# ellip_data_adaptive = data[ (data['cov_type'] == 'ellip') & (data['weight_type'] == 'Adpative') ].sort_values(by=' lambda', ascending=True)
# gt_data = data[ data['cov_type'] == 'GT']
# sphere_data = data[ data['cov_type'] == 'sphere']


# lambda_values = np.array([ 0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3 ,0.5,  1])
# lambda_n = len(lambda_values)

# ellip_picp   = np.array( ellip_data['PICP'] )
# gt_picp      = np.array( [ gt_data['PICP'].values[0] ] * lambda_n )
# sphere_picp  = np.array( [ sphere_data['PICP'].values[0] ] * lambda_n  )

# ellip_volume  = np.array( ellip_data['Volume'] )
# gt_volume     = np.array( [ gt_data['Volume'].values[0] ] * lambda_n )
# sphere_volume = np.array( [ sphere_data['Volume'].values[0] ] * lambda_n )

# ---------------------------------------------------
# 2. Plot for PICP vs. log(lambda)
# ---------------------------------------------------

axis_label_fontsize = 20
title_fontsize =  20
legend_fontsize = 16
tick_fontsize = 20
# print(ellip_data.columns)
plt.figure(figsize=(6, 6))
# color_l = ["#F4A261", "#8ECae6", "#90BE6D"]

T_l = [100,200,300,400,500]
# cmap = plt.get_cmap("viridis")  # You can change this to other colormaps like "plasma", "coolwarm", etc.
# color_l = [cmap(i) for i in np.linspace(0, 1, 5)]
color_l = ["#F6C63C", "#EFA143", "#D96558", "#B43970", "#692F7C"]
# color_l = ["#5E6C82", "#899FB0", "#81B3A9", "#B3C6BB", "#D6CDBE"]

for i in range(len(T_l)):
    T = T_l[i]
    color = color_l[i]
    # print(ellip_data[ ellip_data ['tinit']==T]['PICP'])
    plt.plot( ellip_data_fixed[ ellip_data_fixed ['tinit']==T]['lambda'],
              ellip_data_fixed[ ellip_data_fixed ['tinit']==T]['picp_mean'], marker=',', linestyle ='solid',  linewidth=2.5, alpha = 0.8,
              color = color,label=r"$n$"+"={}".format(T))
    # plt.plot(ellip_data_adaptive[ellip_data_adaptive['tinit'] == T][' lambda'],
    #          ellip_data_adaptive[ellip_data_adaptive['tinit'] == T]['PICP'], marker='x',  linestyle ='dashed',
    #          color=color, label="Adaptive weight, "+r"$T$"+"={}".format(T))
plt.xlabel('λ', fontsize=axis_label_fontsize)
plt.ylabel('Coverage (%)', fontsize=axis_label_fontsize)
plt.title(r'Coverage Rate with $\gamma = {}$'.format(gamma), fontsize=title_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.ylim(86, 97) # <-- Adjust as necessary for your data
plt.legend(fontsize=legend_fontsize)

axis_width = 2.5
ax = plt.gca()
ax.spines['bottom'].set_linewidth(axis_width)  # Bold X-axis
ax.spines['left'].set_linewidth(axis_width)  # Bold Y-axis
ax.spines['top'].set_linewidth(axis_width)  # Bold Top border (optional)
ax.spines['right'].set_linewidth(axis_width)  # Bold Right border (optional)
ax.axhline(y=95, color='grey', linestyle='dotted', linewidth=3)
ax.set_yticks([86,88,90,92,94,95,96,97])


plt.grid(True)
plt.grid(color='gray', linestyle=':', linewidth=0.5)
plt.tight_layout()
plt.savefig('logs/PEMS_adaptive_gamma_{}_picp.png'.format(gamma))
plt.show()

# ---------------------------------------------------
# 3. Plot for Volume vs. log(lambda)
# ---------------------------------------------------
plt.figure(figsize=(6, 6))
for i in range(len(T_l)):
    T = T_l[i]
    color = color_l[i]
    plt.plot( ellip_data_fixed[ ellip_data_fixed ['tinit']==T]['lambda'],
              ellip_data_fixed[ ellip_data_fixed ['tinit']==T]['eff_mean'], marker=',', linestyle ='solid',  linewidth=2.5, alpha = 0.8,
              color = color,label=r"$n$"+"={}".format(T))
    # plt.plot(ellip_data_adaptive[ellip_data_adaptive['tinit'] == T][' lambda'],
    #          ellip_data_adaptive[ellip_data_adaptive['tinit'] == T]['Volume'], marker='x', linestyle ='dashed',
    #          color=color, label="Adaptive weight, "+r"$T$" + "={}".format(T))

# plt.plot(np.log10(lambda_values), gt_volume,     marker='s', label='MultiDimSPCI(GT)')
# plt.plot(np.log(lambda_values), sphere_volume, marker='^', label='Sphere')

plt.xlabel('λ', fontsize=axis_label_fontsize)
plt.ylabel('Efficiency', fontsize=axis_label_fontsize)
plt.title(r'Efficiency with $\gamma = {}$'.format(gamma), fontsize=title_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.ylim(60, 120)  # <-- Adjust as necessary for your data

ax = plt.gca()
ax.spines['bottom'].set_linewidth(axis_width)  # Bold X-axis
ax.spines['left'].set_linewidth(axis_width)  # Bold Y-axis
ax.spines['top'].set_linewidth(axis_width)  # Bold Top border (optional)
ax.spines['right'].set_linewidth(axis_width)  # Bold Right border (optional)

ax.set_ylim(ax.get_ylim()[::-1])
plt.tight_layout()
plt.grid(True)
plt.grid(color='gray', linestyle=':', linewidth=0.5)
plt.savefig('logs/PEMS_adaptive_gamma_{}_volume.png'.format(gamma))
plt.show()

