import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
colors = plt.cm.Blues(np.linspace(0.3, 1, 4))  # Shades of red

x = np.arange(0, 201, 25)

#'a-2_5-0_8-1_2', 'a-5-0_8-1_2', 'a-7_5-0_8-1_2',
#'a-2_5-0_8-1', 'a-5-0_8-1', 'a-7_5-0_8-1',

for color, (guidance, file) in zip(colors, { #'0': 'a0-0_8-1_2', '-2.5': 'a2_5-0_8-1_2', '-5':'a5-0_8-1_2', '-7.5':'a7_5-0_8-1_2', 
           # '0': 'a0-0_8-1', '-2.5': 'a2_5-0_8-1', '-5':'a5-0_8-1', '-7.5':'a7_5-0_8-1'
           #'0': 'a0-0_8-1_2', '2.5':'a-2_5-0_8-1_2', '5': 'a-5-0_8-1_2', '7.5':'a-7_5-0_8-1_2',
            '0': 'a0-0_8-1', '2.5':'a-2_5-0_8-1', '5': 'a-5-0_8-1', '7.5':'a-7_5-0_8-1',
}.items()):

    # Load the CSV file into a DataFrame
    csv_file = f"/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/clf_results/curved_schedule/{file}/evaluation.csv"
    df = pd.read_csv(csv_file)
    
    ax1.plot(x, df['FID'], label=guidance, color=color)

ax1.set_xlabel('Epoch')
ax1.set_ylabel('FID')
ax1.set_title('FID over adaptation over different guidance schedules')
ax1.legend()




# For guidance scheduler
def create_scaled_line(start, end, color, a=0):
    x = np.linspace(0, 1, 201)
    if a == 0:
        y = x
    else:
        y = (np.exp(a * x) - 1) / (np.exp(a) - 1)    
    scaled_y = start + (end - start) * y
    
    ax2.plot(x, scaled_y, color=color, label=a)

create_scaled_line(0.8, 1, color=colors[0], a=0)
create_scaled_line(0.8, 1, color=colors[1], a=2.5)
create_scaled_line(0.8, 1, color=colors[2], a=5)
create_scaled_line(0.8, 1, color=colors[3], a=7.5)

#create_scaled_line(0.8, 1, color=colors[0], a=-0)
#create_scaled_line(0.8, 1, color=colors[1], a=-2.5)
#create_scaled_line(0.8, 1, color=colors[2], a=-5)
#create_scaled_line(0.8, 1, color=colors[3], a=-7.5)

ax2.set_xlabel('Epoch')
ax2.set_ylabel('Guidance scale')
ax2.set_title('Guidance Schedules over training')
ax2.legend()

# Show the plot
plt.tight_layout()
plt.savefig('/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/clf_results/curved_schedule/evaluation0_8-1.png')