
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd


def sigmoid_line(a, b, c, offset):
    # Generate c evenly spaced points between 0 and 1
    x = np.linspace(0, 1, c)
    
    # Apply the sigmoid transformation
    y = 1 / (1 + np.exp(-10 * (x - offset)))  # The factor 10 controls the steepness of the sigmoid

    # Scale the sigmoid output to the range [a, b]
    scaled_y = a + (b - a) * y
    return scaled_y


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
colors = plt.cm.Reds(np.linspace(0.1, 1, 5))  # Shades of red

for idx, (time_weight, file_name) in enumerate({0.1: '0_1', 
    0.3: '0_3', 0.5: '0_5', 0.7: '0_7', 0.9: '0_9'
    }.items()):
    
    sigmoid_values = sigmoid_line(0, 1, 50, time_weight)

    csv_file = f"/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/time_results2/{file_name}/evaluation.csv"
    csv_file = pd.read_csv(csv_file)
    y = csv_file['FID'].min()
    x = [ 0.1, 
    0.3, 0.5, 0.7, 0.9]
    y = [0 if i!=time_weight else y for i in x]
    #x = np.arange(0, 301, 25)
    #ax1.plot(x, y, label=time_weight, color=colors[idx])
    ax1.bar(x, y, label=time_weight, color=colors[idx], width=0.1)
    x = x = np.linspace(0, 1, 50)
    ax2.plot(x, sigmoid_values, label=time_weight, color=colors[idx])

ax1.set_title('FID vs epochs')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('FID')
ax1.legend()

ax2.set_title('Time-step fixed weights')
ax2.set_xlabel('Time-step')
ax2.set_ylabel('Fixed Weight')
ax2.legend()

plt.savefig('/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/time_results2/evaluation.png')
