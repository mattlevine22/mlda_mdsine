import sys

sys.path.append("../")
import torch
from models.runner import Runner
from utils import dict_combiner
import argparse


# use argparse to get command line argument for which experiment to run
parser = argparse.ArgumentParser()
parser.add_argument("--project_name", type=str, default="mdsine2_v4")
parser.add_argument("--fast_dev_run", type=int, default=0)
parser.add_argument("--accelerator", type=str, default="cpu")
parser.add_argument("--devices", type=str, default="auto")
parser.add_argument("--run_all", type=int, default=1)
parser.add_argument("--run_id", type=int, default=0)
args = parser.parse_args()

# unconstrained growth rate
PRETUNED_MECH = "lightning_logs/wpadxclt/checkpoints/last.ckpt"
#lr-A = 1e-13, lr-r = 0.1

# build a dict of experimental conditions
exp_dict = {
    "checkpoint_init": [("pretuned_mech", PRETUNED_MECH)],
    "project_name": [args.project_name],
    "fast_dev_run": [args.fast_dev_run],
    "accelerator": [args.accelerator],
    "devices": [args.devices],
    "normalizer": ["max_min_log10"],
    "loss_name": ["mse"],  # ["mse"],
    "dim_state_mech": [141],  # default is [141] for GLV
    "dim_state_latent": [10, 200],  # default is [0] (no latent),
    "low_bound": [0],  # lowest allowable value of state variables in ODE
    "high_bound": [1e12],  # highest allowable value of state variables in ODE
    "low_bound_latent": [0],  # lowest allowable value of latent variables in ODE
    "high_bound_latent": [1],  # highest allowable value of latent variables in ODE
    "include_control": [True],
    "fully_connected": [True],
    "shared_weights": [False],
    "T_long": [10],
    "use_physics": [True],
    "learn_physics": [False],
    "use_nn_markovian": [True],
    "learn_nn_markovian": [True],
    "use_nn_non_markovian": [False],
    "learn_nn_non_markovian": [False],
    "positive_growth_rate": [False],
    "nn_coefficient_scaling": ["x"],
    "pre_multiply_x": [True],
    "learning_rate": [0.5, 1e-1, 1e-2],
    "layer_width": [600],
    "num_hidden_layers": [4],
    "max_epochs": [60],
    "lr_scheduler_params": [{"patience": 10, "factor": 0.1}],
    "odeint_use_adjoint": [False],
    "odeint_method": ["rk4"],
    "odeint_options": [{"dtype": torch.float32, "step_size": 0.01}],
}

exp_list = dict_combiner(exp_dict)

# Print the length of the experiment list
print("Number of experiments to sweep: ", len(exp_list))

# run the experiment
if args.run_all:
    id_list = list(range(len(exp_list)))
else:
    id_list = [args.run_id]

for i in id_list:
    print("Running experiment ", i+1, " of ", len(exp_list))
    Runner(**exp_list[i])
