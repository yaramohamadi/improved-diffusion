import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



colors = plt.cm.Blues(np.linspace(0.2, 1, 8))  # Shades of red

x = np.arange(0, 201, 25)

csv_file = f"/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/clf_trg_results/fixed_guidance_dataset/data10/evaluation.csv"
df = pd.read_csv(csv_file)

for color, (guidance, file) in zip(colors, {
            #0.8: '0_8', 0.9: '0_9', 0.95: '0_95', 1: '1', 1.05: '1_05', 1.1: '1_1', 1.2: '1_2',
            0:'0', 0.05: '0_05', 0.1: '0_1', 0.2: '0_2'
            # 0:'0', 0.1: '0_1', 0.5: '0_5', 0.25: '0_25', 1:'1', 2.5:'2_5', 5:'5', 10:'10'
}.items()):

    # Load the CSV file into a DataFrame
    # csv_file = f"/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/clf_trg_results/fixed_guidance/{file}/evaluation.csv"
    
    
    if guidance == 0:
        plt.plot(x, df[df['g']==guidance]['FID'], label=guidance, color='red')
    else:
        plt.plot(x, df[df['g']==guidance]['FID'], label=guidance, color=color)

plt.xlabel('Epoch')
plt.ylabel('FID')
plt.title('FID over adaptation over fixed guidances for 10-shot pokemon')
plt.legend()

# Show the plot
plt.tight_layout()
plt.savefig('/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/clf_trg_results/fixed_guidance_dataset/data10/evaluation.png')