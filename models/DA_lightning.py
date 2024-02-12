import os
import numpy as np
from scipy.interpolate import griddata
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from utils import weighted_mse_loss, log_det, neg_log_likelihood_loss
from models.DA_pytorch import DataAssimilator

from pdb import set_trace as bp


def count_examples_left(batch_idx, batch_size, n_examples):
    """Returns number of examples left in batch"""

    # total number of examples processed so far
    n_examples_total = batch_idx * batch_size

    # number of examples left in batch
    n_examples_left = n_examples - n_examples_total

    # if n_examples_left < batch_size, then we have a partial batch
    if n_examples_left < batch_size:
        return max(0, n_examples_left)
    else:
        return batch_size


# Define the pytorch lightning module for training the Simple Encoder model
class DataAssimilatorModule(pl.LightningModule):
    def __init__(
        self,
        dim_obs=1,
        dim_state=10,
        ode=None,
        odeint_use_adjoint=False,
        odeint_method="dopri5",
        odeint_rtol=1e-7,
        odeint_atol=1e-9,
        odeint_options={"dtype": torch.float32},
        use_physics=False,
        use_nn=True,
        nn_coefficient_scaling=1e4,
        low_bound=1e5,
        high_bound=1e12,
        low_bound_latent=0,
        high_bound_latent=1,
        num_hidden_layers=1,
        learn_h=False,
        learn_ObsCov=False,
        learn_StateCov=False,
        da_name="3dvar",
        layer_width=50,
        burnin_frac=0.75,
        learning_rate=0.01,
        activation="gelu",
        monitor_metric="train_loss",
        loss_name="nll",
        lr_scheduler_params={"patience": 3, "factor": 0.5},
        dropout=0.1,
        T_long=1000,
        dt_long=0.1,
        n_examples=2,
        normalizer="inactive",
        normalization_stats=None,
        **kwargs,
    ):
        super(DataAssimilatorModule, self).__init__()

        self.n_examples = n_examples
        self.loss_name = loss_name

        self.dim_obs = dim_obs
        self.dim_state = dim_state
        self.burnin_frac = (
            burnin_frac  # number of burn-in steps to ignore when computing loss
        )

        self.first_forward = (
            True  # for plotting model-related things once at beginnning of training
        )
        self.learning_rate = learning_rate
        self.monitor_metric = monitor_metric
        self.lr_scheduler_params = lr_scheduler_params

        self.low_bound_latent = low_bound_latent
        self.high_bound_latent = high_bound_latent

        # initial condition for the long trajectory
        # make the initial condition 1e7 for the mechanistic variables then 0 for the rest
        self.x0_inv = torch.zeros(1, dim_state)
        self.x0_inv[:, :dim_obs] = 1e7

        # time points for the long trajectory
        self.t_inv = {
            "train": None,
            "val": torch.arange(0, T_long, dt_long),
            "test": torch.arange(0, 10 * T_long, dt_long),
        }

        odeint_params = {
            "use_adjoint": odeint_use_adjoint,
            "method": odeint_method,
            "rtol": odeint_rtol,
            "atol": odeint_atol,
            "options": odeint_options,
        }

        # initialize the model
        self.model = DataAssimilator(
            dim_state=dim_state,
            dim_obs=dim_obs,
            ode=ode,
            odeint_params=odeint_params,
            use_physics=use_physics,
            use_nn=use_nn,
            nn_coefficient_scaling=nn_coefficient_scaling,
            low_bound=low_bound,
            high_bound=high_bound,
            low_bound_latent=low_bound_latent,
            high_bound_latent=high_bound_latent,
            num_hidden_layers=num_hidden_layers,
            layer_width=layer_width,
            dropout=dropout,
            activations=activation,
            learn_h=learn_h,
            learn_ObsCov=learn_ObsCov,
            learn_StateCov=learn_StateCov,
            da_name=da_name,
            normalizer=normalizer,
            normalization_stats=normalization_stats,
        )



    def long_solve(self, device="cpu", stage="val"):
        """This function solves the ODE for a long time, and returns the entire trajectory"""
        # solve the ODE using the initial conditions x0 and time points t
        # solve using default odesolver parameters (atol=1e-7, rtol=1e-9, dopri5)
        x = self.model.solve(
            self.x0_inv, #.to(device),
            self.t_inv[stage], #.to(device),
            params=self.model.odeint_params,
        )
        # x is (N_times, N_batch, dim_state)
        return x

    def forward(self, y_obs, times, controls):

        if self.first_forward:
            self.first_forward = False
            # set device for mech ode
            self.model.rhs.mech_ode.to_device(y_obs.device)
            self.x0_inv = self.x0_inv.to(y_obs)
            self.t_inv = {key: val.to(y_obs.device) if val is not None else None
                    for key, val in self.t_inv.items()}

        # since times currently have the same SPACING across all batches, we can reduce this to just the first batch
        times = times[0].squeeze()

        # TODO: check that times are being used correctly across batches
        # currently, will have different times for each batch, so need batch_size=1.
        y_pred, y_assim, x_pred, x_assim, cov, inv_cov = self.model(
            y_obs=y_obs, times=times
        )
        return y_pred, y_assim, x_pred, x_assim, cov, inv_cov

    def loss(self, y_pred, y_obs, cov, inv_cov, mask=None):
        # y_pred is (N_batch, N_times, dim_obs)
        # y_obs is (N_batch, N_times, dim_obs)

        # The name of the loss function will determine whether we use masking (e.g., l2_mask vs l2)

        # calculate burnin
        n_burnin = int(self.burnin_frac * y_pred.shape[1])

        y_pred = y_pred[:, n_burnin:]
        y_obs = y_obs[:, n_burnin:]
        cov = cov[
            :, n_burnin:
        ]  # this may break 3dvar until we fix the covariance dimensions
        inv_cov = inv_cov[
            :, n_burnin:
        ]  # this may break 3dvar until we fix the covariance dimensions

        # normalize the observations (for other metrics)
        y_obs_normalized = self.model.normalizer.encode(y_obs)
        y_pred_normalized = self.model.normalizer.encode(y_pred)

        # apply mask
        if mask is None:
            mask = torch.ones_like(y_pred)

        mask_sum = torch.sum(mask)
        if mask_sum == 0:
            print("WARNING: mask is all zeros")
            mask_sum = 1

        loss_dict = {}
        for mask_tag in ["", "_masked"]:
            my_mask = mask if mask_tag == "_masked" else torch.ones_like(mask)
            my_mask_sum = torch.sum(my_mask)

            # RMSE between logs (for Travis)...pre-normalization since this is a metric with familiar units
            loss_log_rmse = torch.sqrt(
                torch.sum((my_mask * (torch.log10(y_pred) - torch.log10(y_obs))) ** 2)
                / my_mask_sum
            )

            # relative l2 loss (pointwise)
            # pre-normalization since when using [0,1] normalization, y_obs gets close to 0 and fixates the cost on resolving 0 (i.e. 10^5).
            loss_l2_log_rel = (
                torch.sum(
                    my_mask
                    * (
                        (torch.log10(y_pred) - torch.log10(y_obs)) ** 2
                        / (torch.log10(y_obs) ** 2 + 1e-15)
                    )
                )
                / my_mask_sum
            )

            loss_l2 = (
                torch.sum(my_mask * (y_pred_normalized - y_obs_normalized) ** 2) / my_mask_sum
            )
            # loss_l2 = F.mse_loss(my_mask*y_pred, my_mask*y_obs)
            loss_sup = torch.max(
                torch.abs(my_mask * (y_pred_normalized - y_obs_normalized))
            )

            loss_weighted_mse = weighted_mse_loss(
                my_mask * y_pred_normalized, my_mask * y_obs_normalized, inv_cov
            )

            # TODO: These two need a mask!
            loss_logdet = log_det(cov)
            loss_nll = neg_log_likelihood_loss(
                y_pred_normalized,
                y_obs_normalized,
                cov,
                inv_cov,
            )
            foo_dict = {
                "l2_log_rel": loss_l2_log_rel,
                "mse": loss_l2,
                "l_inf": loss_sup,
                "weighted_mse": loss_weighted_mse,
                "logdet": loss_logdet,
                "nll": loss_nll,
                "log_rmse": loss_log_rmse,
            }

            # append mask_tag to each key
            foo_dict = {f"{key}{mask_tag}": val for key, val in foo_dict.items()}

            # add to loss_dict
            loss_dict.update(foo_dict)

        # check if loss_l2 is nan
        if torch.isnan(loss_l2):
            print("loss_l2 is nan")
            bp()

        return loss_dict

    def training_step(self, batch, batch_idx):
        # print("Training step")
        # bp()
        # print(self.model.rhs.f_nn.state_dict())
        y_obs, x_true, y_true, times, mask, controls, invariant_stats_true = batch
        y_pred, y_assim, x_pred, x_assim, cov, inv_cov = self.forward(
            y_obs, times, controls
        )
        loss_dict = self.loss(y_pred, y_obs, cov, inv_cov, mask)
        for key, val in loss_dict.items():
            self.log(
                f"loss/train/{key}", val, on_step=False, on_epoch=True, prog_bar=True
            )

        n_examples_left = count_examples_left(
            batch_idx, y_obs.shape[0], self.n_examples
        )
        if n_examples_left > 0:
            self.make_batch_figs(
                y_obs,
                x_true,
                y_true,
                times,
                y_pred,
                x_pred,
                x_assim,
                y_assim,
                mask=mask,
                tag=f"Train/idx{batch_idx}",
                n_examples=n_examples_left,
            )

        return loss_dict[self.loss_name]

    def on_after_backward(self):
        self.log_gradient_norms(tag="afterBackward")

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer and its gradient
        # If using mixed precision, the gradients are already unscaled here
        self.log_gradient_norms(tag="beforeOptimizer")
        self.log_parameter_norms(tag="beforeOptimizer")

        wandb.log({"parameters/C_scale": self.model.C_scale.detach()})
        wandb.log({"parameters/Gamma_scale": self.model.Gamma_scale.detach()})
        wandb.log({"parameters/initial_state_mean": self.model.prior_mean.detach()})
        self.log_matrix(self.model.prior_cov.weight.detach(), tag="initial_state_cov")
        self.log_matrix(self.model.Gamma_cov.weight.detach(), tag="Gamma_cov")
        self.log_matrix(self.model.C_cov.weight.detach(), tag="C_cov")
        self.log_matrix(self.model.K.detach(), tag="K")
        self.log_matrix(self.model.h_obs.weight.detach(), tag="h_obs")

    def log_matrix(self, matrix, tag=""):
        # log the learned constant gain K self.model.K.weight.detach()
        param_dict = {}
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                param_dict[f"parameters/{tag}_{i}{j}"] = matrix[i][j]
        wandb.log(param_dict)

    def log_gradient_norms(self, tag=""):
        norm_type = 2.0
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.detach().norm(norm_type)
                name = name.replace(".", "_")
                self.log(
                    f"grad_norm/{tag}/{name}",
                    grad_norm,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

    def log_parameter_norms(self, tag=""):
        norm_type = 2.0
        for name, param in self.named_parameters():
            param_norm = param.detach().norm(norm_type)
            name = name.replace(".", "_")
            self.log(
                f"param_norm/{tag}/{name}",
                param_norm,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

    def validation_step(self, batch, batch_idx):
        y_obs, x_true, y_true, times, mask, controls, invariant_stats_true = batch
        y_pred, y_assim, x_pred, x_assim, cov, inv_cov = self.forward(
            y_obs, times, controls
        )
        # compute the losses
        loss_dict = self.loss(y_pred, y_obs, cov, inv_cov, mask)
        for key, val in loss_dict.items():
            self.log(
                f"loss/val/{key}", val, on_step=False, on_epoch=True, prog_bar=True
            )

        n_examples_left = count_examples_left(
            batch_idx, y_obs.shape[0], self.n_examples
        )
        if n_examples_left > 0:
            # run the model on the long trajectory
            x_long = self.long_solve(device=y_obs.device, stage="val")
            y_long = self.model.h_obs(x_long).detach().cpu().numpy()

            self.make_batch_figs(
                y_obs,
                x_true,
                y_true,
                times,
                y_pred,
                x_pred,
                x_assim,
                y_assim,
                mask=mask,
                y_long=y_long,
                invariant_stats_true=invariant_stats_true,
                tag=f"Val/idx{batch_idx}",
                n_examples=n_examples_left,
            )

        return loss_dict[self.loss_name]

    def make_batch_figs(
        self,
        y_obs,
        x_true,
        y_true,
        times,
        y_pred,
        x_pred,
        x_assim,
        y_assim,
        mask=None,
        y_long=None,
        invariant_stats_true=None,
        tag="",
        n_examples=2,
        yscale="symlog",
    ):
        """This function makes plots for a single batch of data."""

        if n_examples > x_true.shape[0]:
            n_examples = x_true.shape[0]

        for idx in range(n_examples):
            y_obs_idx = y_obs[idx].detach().cpu().numpy()
            y_true_idx = y_true[idx].detach().cpu().numpy()
            x_true_idx = x_true[idx].detach().cpu().numpy()
            times_idx = times[idx].detach().cpu().numpy()
            y_pred_idx = y_pred[idx].detach().cpu().numpy()
            x_pred_idx = x_pred[idx].detach().cpu().numpy()
            x_assim_idx = x_assim[idx].detach().cpu().numpy()
            y_assim_idx = y_assim[idx].detach().cpu().numpy()
            if mask is not None:
                mask_idx = mask[idx].detach().cpu().numpy()
            else:
                mask_idx = np.ones_like(y_obs_idx)

            # First, plot a single temporal trajectory that summarizes the pointwise residuals of the predictions
            plt.figure()
            fig, ax = plt.subplots(
                nrows=1,
                ncols=1,
            )

            mean_error = np.mean(np.log10(y_obs_idx) - np.log10(y_pred_idx), axis=1)
            mask_sum = np.sum(mask_idx, axis=1)

            mask_sum[mask_sum == 0] = 1 # protech against division by zero
            mean_error_masked = (
                np.sum(
                    (mask_idx * (np.log10(y_obs_idx) - np.log10(y_pred_idx))),
                    axis=1,
                )
                / mask_sum
            )

            ax.plot(
                times_idx,
                mean_error,
                label="Unmasked",
            )
            ax.plot(
                times_idx,
                mean_error_masked,
                label="Masked",
            )

            # plot a horizontal line at 0
            ax.axhline(0, color="k", linestyle="--")
            ax.set_xlabel("Time")
            ax.legend()

            # set yscale to [-1.2,1.2] or
            ax.set_ylim([-1.2, 1.2])
            fig.suptitle(
                "Log-Residuals for each patient\n E[log10(x_true(t)) - log10(x_sim(t))](t)"
            )
            wandb.log({f"plots/{tag}/SummaryTraj_{idx}": wandb.Image(fig)})
            plt.close("all")

            # Plot Trajectories
            n_rows = y_obs.shape[-1] + 3

            # set max number of rows to 5
            n_rows = min(n_rows, 5)

            plt.figure()
            fig, axs = plt.subplots(
                nrows=n_rows,
                ncols=3,
                figsize=(30, 6 * n_rows),
                gridspec_kw={"width_ratios": [2.5, 1, 1]},
                # sharex="col",
                squeeze=False,
            )

            # set all y-scales in first two columns to yscale
            # (except for the last row, which is the latent variable)
            for i in range(n_rows - 1):
                # set the x_scale in the third column to yscale
                axs[i, 2].set_xscale(yscale)
                for j in range(2):
                    axs[i, j].set_yscale(yscale)

            # set ylim in last row to [-1.2,1.2]
            axs[-1, 0].set_ylim([self.low_bound_latent-0.2, self.high_bound_latent+0.2])
            axs[-1, 1].set_ylim(
                [self.low_bound_latent - 0.2, self.high_bound_latent + 0.2]
            )

            # master title
            fig.suptitle(
                f"{tag} Trajectories for Index {idx} w/ Predicted Invariant Measure"
            )
            for i in range(n_rows - 3):
                ax = axs[i, 0]
                # plot the assimilated state of the i'th observation
                ax.plot(
                    times_idx,
                    y_assim_idx[:, i],
                    ls="",
                    marker="x",
                    markersize=5,
                    color="black",
                    label="Assimilated",
                )

                # plot the predicted state of the i'th observation
                ax.plot(
                    times_idx,
                    y_pred_idx[:, i],
                    ls="",
                    marker="o",
                    markersize=5,
                    markerfacecolor="none",
                    color="blue",
                    label="Prediction",
                )

                # plot the noisy observations that we are fitting to
                ax.plot(
                    times_idx,
                    y_obs_idx[:, i],
                    ls="",
                    marker="o",
                    markersize=5,
                    alpha=0.5,
                    color="red",
                    label="Observation",
                )

                # plot true state of the i'th observation
                ax.plot(
                    times_idx,
                    y_true_idx[:, i],
                    linewidth=3,
                    color="black",
                    label="Ground Truth",
                )

                ax.set_xlabel("Time")
                ax.set_ylabel(f"Observation {i}")
                ax.set_title(f"Observation for component {i} (Index {idx})")
                if i == 0:
                    ax.legend()

                # in the second column, plot the same as the first column, but only for the last 20 time steps
                ax = axs[i, 1]
                ax.plot(
                    times_idx[-20:],
                    y_assim_idx[-20:, i],
                    ls="",
                    marker="x",
                    markersize=10,
                    color="black",
                    label="Assimilated",
                )
                ax.plot(
                    times_idx[-20:],
                    y_pred_idx[-20:, i],
                    ls="",
                    marker="o",
                    markersize=10,
                    markerfacecolor="none",
                    color="blue",
                    label="Prediction",
                )
                ax.plot(
                    times_idx[-20:],
                    y_obs_idx[-20:, i],
                    ls="",
                    marker="o",
                    markersize=10,
                    alpha=0.5,
                    color="red",
                    label="Observation",
                )
                ax.plot(
                    times_idx[-20:],
                    y_true_idx[-20:, i],
                    linewidth=3,
                    color="black",
                    label="Ground Truth",
                )
                ax.set_xlabel("Time")
                ax.set_ylabel(f"Observation {i}")
                ax.set_title(f"Observation for component {i} (Index {idx})")
                if i == 0:
                    ax.legend()

                # in the third column, plot a kernel density estimate for the distribution of y_long
                if y_long is not None:
                    ax = axs[i, 2]
                    # use seaborn kdeplot
                    sns.kdeplot(
                        y_long[..., i].squeeze(),
                        ax=ax,
                        fill=True,
                        color="blue",
                        label="Predicted",
                    )
                    ax.set_xlabel(f"Observation {i}")
                    ax.set_ylabel("Density")
                    ax.set_title(f"Invariant Distribution for Observation {i}")
                    ax.legend()

                # warning that we are not plotting the true invariant distribution
                print("WARNING: true invariant distribution not plotted.")

                # if invariant_stats_true is not None and len(invariant_stats_true) > 0:
                #     # the [0] takes the first batch (all batches are the same, though, for this piece of data)
                #     ax.plot(
                #         invariant_stats_true["kde_list"][i]["x_grid"][0].detach().cpu(),
                #         invariant_stats_true["kde_list"][i]["p_x"][0].detach().cpu(),
                #         label="Reference",
                #         color="black",
                #     )
                #     ax.legend()

            # plot all assimilated/predicted variables in observation
            ax = axs[-2, 0]
            ax.plot(
                times_idx,
                y_assim_idx,
                ls="",
                marker="x",
                markersize=10,
                color="gray",
                label="Assimilated",
            )
            ax.plot(
                times_idx,
                y_pred_idx,
                ls="",
                marker="o",
                markersize=10,
                markerfacecolor="none",
                color="gray",
                label="Prediction",
            )
            ax.set_title(f"All learned observed variables")

            ax = axs[-2, 1]
            ax.plot(
                times_idx[-20:],
                y_assim_idx[-20:],
                ls="",
                marker="x",
                markersize=10,
                color="gray",
                label="Assimilated",
            )
            ax.plot(
                times_idx[-20:],
                y_pred_idx[-20:],
                ls="",
                marker="o",
                markersize=10,
                markerfacecolor="none",
                color="gray",
                label="Prediction",
            )
            ax.set_title(f"All learned observed variables")

            # plot all assimilated/predicted variables in observation
            D = self.dim_obs
            ax = axs[-1, 0]
            ax.plot(
                times_idx,
                x_assim_idx[:, D:],
                ls="",
                marker="x",
                markersize=10,
                color="gray",
                label="Assimilated",
            )
            ax.plot(
                times_idx,
                x_pred_idx[:, D:],
                ls="",
                marker="o",
                markersize=10,
                markerfacecolor="none",
                color="gray",
                label="Prediction",
            )
            ax.set_title(f"All learned latent variables")

            ax = axs[-1, 1]
            ax.plot(
                times_idx[-20:],
                x_assim_idx[-20:, D:],
                ls="",
                marker="x",
                markersize=10,
                color="gray",
                label="Assimilated",
            )
            ax.plot(
                times_idx[-20:],
                x_pred_idx[-20:, D:],
                ls="",
                marker="o",
                markersize=10,
                markerfacecolor="none",
                color="gray",
                label="Prediction",
            )
            ax.set_title(f"All learned latent variables")

            # plot the true latent variables
            ax = axs[-3, 0]
            ax.plot(
                times_idx, y_true_idx, linewidth=3, color="gray", label="Ground Truth"
            )
            ax.set_title(f"True Observed Variables")
            ax.set_xlabel("Time")

            ax = axs[-3, 1]
            ax.plot(
                times_idx[-20:],
                y_true_idx[-20:],
                linewidth=3,
                color="gray",
                label="Ground Truth",
            )
            ax.set_title(f"True Observed Variables")
            ax.set_xlabel("Time")

            plt.subplots_adjust(hspace=0.5)
            wandb.log({f"plots/{tag}/Trajectories_{idx}": wandb.Image(fig)})
            plt.close("all")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        dt = self.trainer.datamodule.test_sample_rates[dataloader_idx]
        y_obs, x_true, y_true, times, mask, controls, invariant_stats_true = batch

        y_pred, y_assim, x_pred, x_assim, cov, inv_cov = self.forward(
            y_obs, times, controls
        )

        # compute the losses
        loss_dict = self.loss(y_pred, y_obs, cov, inv_cov, mask)
        for key, val in loss_dict.items():
            self.log(
                f"loss/test/{key}/dt{dt}",
                val,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        n_examples_left = count_examples_left(
            batch_idx, y_obs.shape[0], self.n_examples
        )

        # log plots
        if n_examples_left > 0:
            # run the model on the long trajectory
            x_long = self.long_solve(device=y_obs.device, stage="test")
            y_long = self.model.h_obs(x_long).detach().cpu().numpy()
            self.make_batch_figs(
                y_obs,
                x_true,
                y_true,
                times,
                y_pred,
                x_pred,
                x_assim,
                y_assim,
                mask=mask,
                y_long=y_long,
                invariant_stats_true=invariant_stats_true,
                tag=f"Test/idx{batch_idx}/dt{dt}",
                n_examples=n_examples_left,
            )

        return loss_dict[self.loss_name]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        config = {
            # REQUIRED: The scheduler instance
            "scheduler": ReduceLROnPlateau(
                optimizer, verbose=True, **self.lr_scheduler_params
            ),
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": self.monitor_metric,  # "val_loss",
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            "strict": True,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": None,
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": config,
        }
