import numpy as np
import matplotlib.pyplot as plt
import glob

# >>> x.keys()
# dict_keys(['train', 'val'])
# >>> x['train'].keys()
# dict_keys(["2", "3"])
# >>> x['train']['2'].keys()
# dict_keys(['times', 'y_obs', 'y_pred'])
# >>> x['train']['2']['y_obs'].shape
# (1, 75, 141)

perturb_times = {
    "High Fat Diet": (21.5000, 28.5000),
    "Vancomycin": (35.5000, 42.5000),
    "Gentamicin": (50.5000, 57.5000),
}

main_path = 'lightning_logs'
dir_dict = {
    # "GLV nominal": "abc",
    "GLV tuned": "wpadxclt",
    "GLV tuned 2": "wpadxclt"
}

# build a dictionary of relevant data
data_dict = {}
for name, dir_name in dir_dict.items():
    foo_path = f"{main_path}/{dir_name}/checkpoints/predictions_*"
    # build a list of all files that match the pattern
    files = glob.glob(foo_path)
    # select the file with the highest epoch number
    file = max(files, key=lambda x: int(x.split('epoch')[1].split('.')[0]))
    # load the file
    x = np.load(file, allow_pickle=True).item()
    # store the relevant data
    data_dict[name] = x

def mae_per_time(y_obs, y_pred):
    return np.mean(np.abs(np.log10(y_obs) - np.log10(y_pred)), axis=-1)


# First, plot the mean error per time for each
# save each plot as a separate file
fig, axs = plt.subplots(4, 1, figsize=(10, 20))
ax_idx = 0  # Index to track which subplot axis to use
for stage in ["train", "val"]:
    id_list = data_dict["GLV tuned"][stage]
    for idx in id_list:
        ax = axs[ax_idx]  # Use the current axis for plotting
        for name, data in data_dict.items():
            y_obs = data[stage][idx]["y_obs"].squeeze()
            y_pred = data[stage][idx]["y_pred"].squeeze()
            mae = mae_per_time(y_obs, y_pred).squeeze()
            ax.plot(data[stage][idx]["times"].squeeze(), mae, label=name)
        ax.set_title(f"{stage} Mouse-{idx}: Mean Absolute Error")
        ax.set_xlabel("Time")
        # create shaded regions for perturbations with different colors and labels for each
        for i, (perturb, times) in enumerate(perturb_times.items()):
            ax.axvspan(*times, color=f"C{i}", alpha=0.3, label=perturb)
        ax.legend()

        ax_idx += 1  # Move to the next subplot axis

plt.tight_layout()  # Adjust layout to prevent overlap
fig.savefig("mae_per_time.png")  # Save the figure to a file
