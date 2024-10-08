
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
colors = plt.cm.Reds(np.linspace(0.1, 1, 5))  # Shades of red

for idx, (g, g_name) in enumerate({
    0.25: '0_25-1', 0.5: '0_5-1', 0.75: '0_75-1', 0.9: '0_9-1'
    }.items()):
    
    guidance_scale = np.linspace(1, g, 50)

    csv_file = f"/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/clf_results/time_linear_guidance/{g_name}/evaluation.csv"
    csv_file = pd.read_csv(csv_file)
    y = csv_file['FID'].min()
    x = [ 0.25, 
    0.5, 0.75, 0.9]
    y = [0 if i!=g else y for i in x]
    #x = np.arange(0, 301, 25)
    #ax1.plot(x, y, label=time_weight, color=colors[idx])
    ax1.bar(x, y, label=g, color=colors[idx], width=0.1)
    x = np.linspace(0, 50, 50)
    ax2.plot(x, guidance_scale, label=g, color=colors[idx])

ax1.set_title('FID vs Timestep')
ax1.set_xlabel('Timestep')
ax1.set_ylabel('FID')
ax1.legend()
ax1.set_ylim([0, 140])

ax2.set_title('Time-based linear schedule')
ax2.set_xlabel('Time-step')
ax2.set_ylabel('Linear Schedule')

ax2.legend()

plt.savefig('/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/clf_results/time_linear_guidance/evaluation.png')
