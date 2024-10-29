import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


colors = plt.cm.Blues(np.linspace(0.2, 1, 8))  # Shades of red

x = np.arange(0, 201, 25)



modes = ['a3ft', 'attention_finetune', 'finetune']
metric = 'FID'

for mode in modes: 
    csv_file = f"/home/ymbahram/scratch/baselines/a3ft/results_samesample/data10/{mode}/{metric}.csv"
    df = pd.read_csv(csv_file)

    plt.plot(x, df[metric], label=mode)


plt.xlabel('Epoch')
plt.ylabel(metric)
plt.title('a3ft over fixed guidances for data10-shot pokemon')
plt.legend()
# Show the plot
plt.tight_layout()
plt.savefig(f'/home/ymbahram/scratch/baselines/a3ft/results_samesample/data10/{metric}.png')