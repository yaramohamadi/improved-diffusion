import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


colors = plt.cm.Blues(np.linspace(0.2, 1, 8))  # Shades of red


modes = ['a3ft',  'finetune'] # 'attention_finetune',
metric = 'KID'

for mode in modes: 

    plt.figure(figsize=(6, 4))

    for data in ['data10', 'data100', 'data700', 'data2503']:
        if data == 'data10':
           csv_file = f"/home/ymbahram/scratch/baselines/a3ft/results_samesample/{data}/{mode}/{metric}.csv"
        else:
            csv_file = f"/home/ymbahram/scratch/baselines/a3ft/results_samesample/{data}/{mode}/FID_KID.csv"
        df = pd.read_csv(csv_file)

        x = np.arange(0, len(df[metric])*25, 25)
        plt.plot(x, df[metric], label=f"{mode}-{data}")


    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.title(f'{mode} over 0 guidances for pokemon')
    plt.legend()
    plt.ylim([0, 0.3])
    # Show the plot
    plt.tight_layout()
    plt.savefig(f'/home/ymbahram/scratch/baselines/a3ft/results_samesample/{metric}_{mode}_over_data.png')
    plt.show()