import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


colors = plt.cm.Blues(np.linspace(0.2, 1, 8))  # Shades of red

x = np.arange(0, 201, 25)

csv_file = f"/home/ymbahram/scratch/clf_trg_results/results_samesample/time-step_g-only/data10/evaluation.csv" # Only KID
df = pd.read_csv(csv_file)

g = 0
df = df[df['g']==g] # select for plot

for color, p2_gamma in zip(colors, [0#, 0.1, 0.3, 1, 3, 10
                                    ]):

    if p2_gamma == 0:
        plt.plot(x, df[df['p2_gamma']==0]['KID'], label=p2_gamma, color='red')
    else:
        plt.plot(x, df[df['p2_gamma']==p2_gamma]['KID'], label=p2_gamma, color=color)


plt.xlabel('Epoch')
plt.ylabel('KID')
plt.title(f'KID P2_gamma (g only {g}) Schedule adaptation over fixed guidances for data10-shot pokemon')
plt.legend()
plt.ylim([0,0.5])
# Show the plot
plt.tight_layout()
plt.savefig(f'/home/ymbahram/scratch/clf_trg_results/results_samesample/time-step_g-only/data10/KID_{g}.png')