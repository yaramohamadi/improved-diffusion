import numpy as np
import matplotlib.pyplot as plt
from improved_diffusion import gaussian_diffusion as gd

def compute_and_plot_snr_curves(noise_schedule, steps, timestep_respacing, p2_gamma_values):
    def space_timesteps(num_timesteps, section_counts):
        if isinstance(section_counts, str):
            if section_counts.startswith("ddim"):
                desired_count = int(section_counts[len("ddim"):])
                for i in range(1, num_timesteps):
                    if len(range(0, num_timesteps, i)) == desired_count:
                        return set(range(0, num_timesteps, i))
                raise ValueError(
                    f"cannot create exactly {num_timesteps} steps with an integer stride"
                )
            section_counts = [int(x) for x in section_counts.split(",")]
        size_per = num_timesteps // len(section_counts)
        extra = num_timesteps % len(section_counts)
        start_idx = 0
        all_steps = []
        for i, section_count in enumerate(section_counts):
            size = size_per + (1 if i < extra else 0)
            if size < section_count:
                raise ValueError(
                    f"cannot divide section of {size} steps into {section_count}"
                )
            if section_count <= 1:
                frac_stride = 1
            else:
                frac_stride = (size - 1) / (section_count - 1)
            cur_idx = 0.0
            taken_steps = []
            for _ in range(section_count):
                taken_steps.append(start_idx + round(cur_idx))
                cur_idx += frac_stride
            all_steps += taken_steps
            start_idx += size
        return set(all_steps)

    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    betas = np.array(betas, dtype=np.float64)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    use_timesteps = set(space_timesteps(steps, timestep_respacing))
    timestep_map = []
    original_num_steps = len(betas)

    last_alpha_cumprod = 1.0
    new_betas = []
    for i, alpha_cumprod in enumerate(alphas_cumprod):
        if i in use_timesteps:
            new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
            last_alpha_cumprod = alpha_cumprod
            timestep_map.append(i)
    betas = np.array(new_betas)

    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    snr = 1.0 / (1 - alphas_cumprod) - 1

    plt.figure(figsize=(10, 6))
    for p2_gamma in p2_gamma_values:
        curve = 1 - 1 / (1 + snr)**p2_gamma
        plt.plot(curve, label=f"p2_gamma = {p2_gamma}")

    plt.title("SNR Curves for Different p2_gamma Values")
    plt.xlabel("Timesteps")
    plt.ylabel("1 / (1 + SNR)^p2_gamma")
    plt.legend()
    plt.grid()
    plt.savefig("/home/ens/AT74470/clf_results/clf_xs_xt/p2_weighting_linear.png")

# Example usage
noise_schedule = "linear"
steps = 4000
timestep_respacing = "ddim50"
p2_gamma_values = [0.1, 0.5, 2.0, 10, 50]

compute_and_plot_snr_curves(noise_schedule, steps, timestep_respacing, p2_gamma_values)