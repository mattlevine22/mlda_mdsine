import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint
import pytorch_lightning as pl
from sklearn.utils import shuffle as skshuffle
import mdsine2 as md2
import random

from pdb import set_trace as bp

# Define the paths to the data
# FILTERED_FNAME = "/Users/levinema/code_projects/MDSINE2_Paper/datasets/gibson/preprocessed/gibson_healthy_agg_filtered.pkl"
# PERT_FNAME = "/Users/levinema/code_projects/MDSINE2_Paper/datasets/gibson/raw_tables/perturbations.tsv"
# n_subsample = 1
# SIMS_FNAME = f"/Users/levinema/code_projects/mdsine_local/data/solutions_custom/subsample{n_subsample}"

FILTERED_FNAME = "../data/gibson_healthy_agg_filtered.pkl"
PERT_FNAME = "../data/perturbations.tsv"
n_subsample = 1
SIMS_FNAME = f"../data/solutions_custom/subsample{n_subsample}"

# PARAM_DIR = "/Users/levinema/code_projects/MDSINE2/tmp/"
PARAM_DIR = os.path.expanduser("~/code_projects/MDSINE2/tmp/")

def load_dyn_sys_class(dataset_name):
    dataset_classes = {
        "GLV": GeneralizedLotkaVolterra,
        # Add more dataset classes here for other systems
    }

    if dataset_name in dataset_classes:
        return dataset_classes[dataset_name]
    else:
        raise ValueError(f"Dataset class '{dataset_name}' not found.")


class DynSys(object):
    def __init__(self, state_dim=1, obs_noise_std=1, obs_inds=[0]):
        self.state_dim = state_dim
        self.obs_noise_std = obs_noise_std
        self.obs_inds = obs_inds

    def rhs(self, t, x):
        raise NotImplementedError

    def get_inits(self, size):
        raise NotImplementedError

    def solve(self, N_traj, T, dt):
        """ode solver for the dynamical system.
        Returns xyz, times, where:
        xyz is a tensor of shape (N_traj, N_times, state_dim)
        times is a tensor of shape (N_times, 1)
        """
        times = torch.arange(0, T, dt)
        xyz0 = self.get_inits(N_traj)
        xyz = odeint(self.rhs, xyz0, times)
        return xyz.permute(1, 0, 2), times.reshape(-1, 1)

    def h_obs(self, x):
        """observation function: default is to observe the first component of the state"""
        return x[..., self.obs_inds]

    def noisy_obs(self, x):
        """default is to apply additive i.i.d. zero-mean Gaussian noise to observations"""
        y = self.h_obs(x)
        y_noisy = y + self.obs_noise_std * torch.randn_like(y)
        return y_noisy


class GeneralizedLotkaVolterra(DynSys):
    def __init__(
        self,
        patient_id="2",  # only used to choose which mechanistic parameter set is best (also loads perturbations, but these are the same across patients currently)
        param_dir=PARAM_DIR,
    ):
        super(GeneralizedLotkaVolterra, self).__init__()

        # load data object
        data = MDData()

        self.param_dir = param_dir
        self.threshold = 1e15
        self.close_to_zero = 1e-2

        # load perturbations for single patient (same for all subjects so far)
        # Todo: will need to deal with this when we have different perturbations for different subjects
        self.perturbations, self.times = data.load_perturbations(patient_id=patient_id)

        idx, _, _, _ = data.get_best_sim(patient_id=patient_id)
        self.load_params_from_file(idx)

    def load_params_from_file(self, iter=-1):
        print("Loading parameters from file")

        interactions = np.load(self.param_dir + "interactions.npy")
        print("Number of MCMC iterations: ", interactions.shape[0])
        interactions = interactions[iter]
        growth = np.load(self.param_dir + "growth.npy")[iter]

        pert_params = np.load(self.param_dir + "perturbations.npz")

        self.state_dim = interactions.shape[0]

        # load as float tensors
        self.r = torch.from_numpy(growth).float()
        self.A = torch.from_numpy(interactions).float()
        self.pert_params = {
            k: torch.from_numpy(v[iter]).float() for k, v in pert_params.items()
        }

    def to_device(self, device):
        """Move all tensors to the specified device."""
        self.r = self.r.to(device)
        self.A = self.A.to(device)
        self.pert_params = {k: v.to(device) for k, v in self.pert_params.items()}

    def rhs(self, t, x):
        # Generalized Lotka-Volterra equations
        # dx/dt = x * [r * ( 1 + gamma(t) ) + Ax]
        # the first term is element-wise multiplication of state x and growth rate r

        # # keep x above a threshold to avoid numerical issues
        # x[x < self.close_to_zero] = self.close_to_zero

        perturb = torch.zeros_like(x)
        for key, val in self.pert_params.items():
            if t >= self.times[key][0].squeeze() and t <= self.times[key][1].squeeze():
                # sub_df = self.perturbations[self.perturbations.name == key]
                # t_start = torch.from_numpy(sub_df["start"].values).float()
                # t_end = torch.from_numpy(sub_df["end"].values).float()
                # if t >= t_start and t <= t_end:
                perturb += val

        # x: (N_batch, N_states)
        # r: (N_states, )
        # A: (N_states, N_states)
        # perturb: (N_batch, N_states)
        # dx: (N_batch, N_states)
        dx = x * (self.r * (1 + perturb) + torch.matmul(self.A, x.T).T)
        return dx


class MDData(object):
    def __init__(self, fname=FILTERED_FNAME, sims_name=SIMS_FNAME):
        self.fname = fname
        self.sims_name = sims_name
        self.x = md2.Study.load(fname)
        self.limit_of_detection = 1e5

    def get_patient(self, patient_id):
        t = torch.from_numpy(self.x[str(patient_id)].times).float()
        x = torch.from_numpy(self.x[str(patient_id)].matrix()["abs"]).float()

        # create a mask for the indices to keep (i.e., when x is not zero)
        mask = x != 0

        x[x < self.limit_of_detection] = self.limit_of_detection
        return t, x, mask

    def get_patient_init(self, patient_id="2"):
        _, x, _ = self.get_patient(patient_id)

        return x[:, 0]

    def load_perturbations(self, patient_id="2", fname=PERT_FNAME):
        df = pd.read_csv(fname, sep="\t")
        df = df[df["subject"] == int(patient_id)]
        self.times = {}
        for key in df.name:
            sub_df = df[df.name == key]
            t_start = torch.from_numpy(sub_df["start"].values).float()
            t_end = torch.from_numpy(sub_df["end"].values).float()
            self.times[key] = (t_start, t_end)
        return df, self.times

    def get_best_sim(self, patient_id="2"):
        # load the mcmc sims
        sims = np.load(f"{self.sims_name}/subject{patient_id}.npy")
        sim_times = np.load(f"{self.sims_name}/subject{patient_id}_times.npy")
        iters = np.load(f"{self.sims_name}/subject{patient_id}_indices.npy")
        # sims is (n_mcmc_iters, n_states, )

        # threshold the solution to the limit of detection
        sims[sims < self.limit_of_detection] = self.limit_of_detection

        # get the patient data
        _, x, mask = self.get_patient(patient_id)

        my_eps = 1e-16
        log_sims = np.log10(sims + my_eps)
        log_x = np.log10(x.numpy() + my_eps)
        # compute MSE between x and each sim
        mse = np.mean((log_sims - log_x) ** 2, axis=(1, 2))

        # identify index of best sim
        ind = int(np.argmin(mse))
        # print("best sample index", ind)

        best_sim_idx = iters[ind]
        best_sim = sims[ind]
        # print("best MCMC index", best_sim_idx)

        return best_sim_idx, best_sim, sim_times, x.numpy()


class MDDataset(Dataset):
    def __init__(self, patients):
        self.patients = patients
        self.data = MDData()  # Assuming MDData class is defined in a previous cell
        self.invariant_stats_true = []
        self.compute_invariant_stats()

    def __len__(self):
        return len(self.patients)

    # compute summary statistics (mean, std, log10(mean), etc) over all patients
    def compute_invariant_stats(self):
        x_all = []
        for patient_id in self.patients:
            _, x, mask = self.data.get_patient(patient_id)
            x_all.append(x)
        x_all = torch.cat(x_all, dim=1)

        # self.invariant_stats_true = {
        #     "max": torch.max(x_all, dim=1).values,
        #     "min": torch.min(x_all, dim=1).values,
        #     "max_log10": torch.max(torch.log10(x_all), dim=1).values,
        #     "min_log10": torch.min(torch.log10(x_all), dim=1).values,
        #     "mean": torch.mean(x_all, dim=1),
        #     "std": torch.std(x_all, dim=1),
        #     "mean_log10": torch.mean(torch.log10(x_all), dim=1),
        #     "std_log10": torch.std(torch.log10(x_all), dim=1),
        # }

        # compute summary statistics over all patients and states to get scalars
        # since all states are similar, maybe this is better than normalizing each state separately?
        # I changed this because some states change very little in the dataset, so the std/range is very small
        # ...then, when exploring growing that state, the effective value
        self.invariant_stats_true = {
            "max": torch.max(x_all),
            "min": torch.min(x_all),
            "max_log10": torch.max(torch.log10(x_all)),
            "min_log10": torch.min(torch.log10(x_all)),
            "mean": torch.mean(x_all),
            "std": torch.std(x_all),
            "mean_log10": torch.mean(torch.log10(x_all)),
            "std_log10": torch.std(torch.log10(x_all)),
        }

    def __getitem__(self, idx):
        patient_id = self.patients[idx]
        t, x, mask = self.data.get_patient(patient_id)
        perturbation = []
        # perturbation = self.data.load_perturbations(patient_id)

        # make x: (n_times, n_states)
        x = x.T
        y_obs_batch = x
        x_true_batch = x
        y_true_batch = x
        times_batch = t

        # make mask: (n_times, n_states)
        mask_batch = mask.T

        controls_batch = perturbation

        return (
            y_obs_batch,
            x_true_batch,
            y_true_batch,
            times_batch,
            mask_batch,
            controls_batch,
            self.invariant_stats_true,
        )


class DynamicsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        shuffle="every_epoch",  # can be 'once', 'every_epoch', or 'never'
        inds={"train": ["2", "3"], "val": ["4", "5"], "test": ["4", "5"]},
        batch_size=64,
        train_sample_rate=0.01,
        test_sample_rates=[0.01],
        obs_noise_std=1,
        obs_inds=[0],
        ode_params={},
        dyn_sys_name="GLV",
        **kwargs,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.inds = inds
        self.train_sample_rate = train_sample_rate
        self.test_sample_rates = test_sample_rates
        self.ode_params = ode_params
        self.obs_inds = obs_inds
        self.obs_noise_std = obs_noise_std
        self.dyn_sys_name = dyn_sys_name
        self.shuffle = shuffle

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        self.train = MDDataset(patients=self.inds["train"])

        self.val = MDDataset(patients=self.inds["val"])

        # build a dictionary of test datasets with different sample rates
        self.test = MDDataset(patients=self.inds["test"])

        # use the same normalization statistics for train, val, and test
        self.train.normalization_stats = self.train.invariant_stats_true
        self.val.normalization_stats = self.train.invariant_stats_true
        self.test.normalization_stats = self.train.invariant_stats_true

    def get_dataloader(self, data):
        if self.shuffle == "once":
            shuffle = False
            data = skshuffle(data)
        elif self.shuffle == "every_epoch":
            shuffle = True
        else:
            shuffle = False
        return DataLoader(data, batch_size=self.batch_size, shuffle=shuffle)

    def train_dataloader(self):
        return self.get_dataloader(self.train)

    def val_dataloader(self):
        return self.get_dataloader(self.val)

    def test_dataloader(self):
        return self.get_dataloader(self.test)
