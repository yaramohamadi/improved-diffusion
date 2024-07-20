import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



colors = plt.cm.Blues(np.linspace(0.2, 1, 8))  # Shades of red

x = np.arange(0, 201, 25)

for color, (guidance, file) in zip(colors, {
            0.8: '0_8', 0.9: '0_9', 0.95: '0_95', 1: '1', 1.05: '1_05', 1.1: '1_1', 1.2: '1_2',
}.items()):

    # Load the CSV file into a DataFrame
    csv_file = f"/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/clf_results/fixed_guidance/{file}/evaluation.csv"
    df = pd.read_csv(csv_file)
    
    if guidance == 1:
        plt.plot(x, df['FID'], label=guidance, color='red')
    else:
        plt.plot(x, df['FID'], label=guidance, color=color)

plt.xlabel('Epoch')
plt.ylabel('FID')
plt.title('FID over adaptation over different guidance schedules')
plt.legend()

# Show the plot
plt.tight_layout()
plt.savefig('/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/clf_results/fixed_guidance/evaluation.png')