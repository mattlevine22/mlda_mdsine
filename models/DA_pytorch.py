import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from models.strNN import StrNN
from utils import get_activation, odeint_wrapper, Symmetric, MatrixExponential
from utils import batch_covariance
from utils import load_normalizer_class

from pdb import set_trace as bp

torch.autograd.set_detect_anomaly(True)


class DataAssimilator(nn.Module):

    def __init__(
        self,
        dim_state_mech: int,
        dim_state_latent: int,
        dim_obs: int,
        da_name: str = "3dvar",  # "enkf" or "3dvar"
        N_ensemble: int = 10,
        ode: object = None,
        odeint_params: dict = {
            "use_adjoint": False,
            "method": "dopri5",
            "rtol": 1e-7,
            "atol": 1e-9,
            "options": {"dtype": torch.float32},
        },
        use_physics: bool = False,
        use_nn_markovian: bool = True,
        use_nn_non_markovian: bool = False,
        learn_physics: bool = False,
        learn_nn_markovian: bool = False,
        learn_nn_non_markovian: bool = False,
        learn_h: bool = False,
        learn_ObsCov: bool = False,
        learn_StateCov: bool = False,
        learn_prior_mean: bool = False,
        learn_prior_cov: bool = False,
        nn_coefficient_scaling: float = 1e3,
        pre_multiply_x: bool = True,
        low_bound: float = 1e5,
        high_bound: float = 1e12,
        low_bound_latent: float = 0,
        high_bound_latent: float = 1,
        num_hidden_layers: int = 1,
        layer_width: int = 50,
        activations: int = "gelu",
        dropout: float = 0.01,
        include_control: bool = True,
        fully_connected: bool = True,
        shared_weights: bool = False,
        normalizer: str = "MaxMinLog10Normalizer",
        normalization_stats=None,
    ):
        super(DataAssimilator, self).__init__()

        # set up normalizer
        self.normalizer = load_normalizer_class(normalizer)(
            normalization_stats=normalization_stats
        )

        dim_state = dim_state_mech + dim_state_latent
        self.da_name = da_name
        self.N_ensemble = N_ensemble
        self.dim_state = dim_state
        self.dim_obs = dim_obs
        self.odeint_params = odeint_params

        self.rhs = HybridODE(
            dim_state_mech=dim_state_mech,
            dim_state_latent=dim_state_latent,
            normalizer=self.normalizer,
            ode=ode,
            use_physics=use_physics,
            use_nn_markovian=use_nn_markovian,
            use_nn_non_markovian=use_nn_non_markovian,
            num_hidden_layers=num_hidden_layers,
            layer_width=layer_width,
            activations=activations,
            dropout=dropout,
            nn_coefficient_scaling=nn_coefficient_scaling,
            pre_multiply_x=pre_multiply_x,
            low_bound=low_bound,
            high_bound=high_bound,
            low_bound_latent=low_bound_latent,
            high_bound_latent=high_bound_latent,
            include_control=include_control,
            fully_connected=fully_connected,
            shared_weights=shared_weights,
        )

        # initialize the observation map to be an unbiased identity map
        self.h_obs = nn.Linear(dim_state, dim_obs, bias=True)
        # set to identity of shape (dim_obs, dim_state)
        self.h_obs.weight.data = torch.eye(dim_obs, dim_state)
        self.h_obs.bias.data = torch.zeros(dim_obs)

        self.Gamma_cov = nn.Linear(dim_obs, dim_obs, bias=False)
        if learn_ObsCov:
            self.Gamma_cov.weight.data = torch.zeros(dim_obs) + 0.01 * torch.randn(
                dim_obs, dim_obs
            )
            # torch.abs(self.Gamma_cov.weight.data)
            parametrize.register_parametrization(self.Gamma_cov, "weight", Symmetric())
            parametrize.register_parametrization(
                self.Gamma_cov, "weight", MatrixExponential()
            )
        else:
            self.Gamma_cov.weight.data = torch.eye(dim_obs, dim_obs)

        print("Initial Gamma_cov: ", self.Gamma_cov.weight.data)

        # initialize the state noise covariance to be 0.1*identity
        self.C_cov = nn.Linear(dim_state, dim_state, bias=False)
        if learn_StateCov:
            self.C_cov.weight.data = torch.zeros(dim_state) + 0.01 * torch.randn(
                dim_state, dim_state
            )
            parametrize.register_parametrization(self.C_cov, "weight", Symmetric())
            parametrize.register_parametrization(
                self.C_cov, "weight", MatrixExponential()
            )
        else:
            self.C_cov.weight.data = torch.eye(dim_state, dim_state)

        print("Initial C_cov: ", self.C_cov.weight.data)

        # create scale parameters in SDTDEV units to hopefully make learning easier
        self.Gamma_scale = nn.Parameter(
            torch.tensor(0.1), requires_grad=learn_ObsCov
        )  # this is the scale of the observation noise STDEV
        self.C_scale = nn.Parameter(
            torch.tensor(0.1), requires_grad=learn_StateCov
        )  # this is the scale of the state noise STDEV

        self.compute_constant_K()
        print("Initial K: ", self.K)

        # set an initial condition for the state and register it as a buffer
        # note that this is better than self.x0 = x0 because pytorch-lightning will manage the device
        # so you don't have to do .to(device) every time you use it
        # Note: this initial condition gets learned under 3dvar or enkf but not under no_assim
        self.register_buffer("x0", torch.zeros(dim_state, requires_grad=True))

        # create a learnable prior mean and covariance to draw EnKF ensemble members from
        self.prior_mean = nn.Parameter(torch.zeros(dim_state))
        self.prior_cov = nn.Linear(dim_state, dim_state, bias=False)
        self.prior_cov.weight.data = torch.zeros(dim_state) + 0.01 * torch.randn(
            dim_state, dim_state
        )
        parametrize.register_parametrization(self.prior_cov, "weight", Symmetric())
        parametrize.register_parametrization(
            self.prior_cov, "weight", MatrixExponential()
        )

        # set learnability of parameters
        self.set_learnability(
            learn_prior_mean, learn_prior_cov,learn_h, learn_ObsCov, learn_StateCov,
            learn_physics, learn_nn_markovian, learn_nn_non_markovian)

    def set_learnability(
        self,
        learn_prior_mean,
        learn_prior_cov,
        learn_h,
        learn_ObsCov,
        learn_StateCov,
        learn_physics,
        learn_nn_markovian,
        learn_nn_non_markovian,
    ):

        self.prior_mean.requires_grad = learn_prior_mean
        for param in self.prior_cov.parameters():
            param.requires_grad = learn_prior_cov
        for param in self.h_obs.parameters():
            param.requires_grad = learn_h
        for param in self.Gamma_cov.parameters():
            param.requires_grad = learn_ObsCov
        for param in self.C_cov.parameters():
            param.requires_grad = learn_StateCov
        for param in self.rhs.f_nn_markovian.parameters():
            param.requires_grad = learn_nn_markovian
        for param in self.rhs.f_nn_non_markovian.parameters():
            param.requires_grad = learn_nn_non_markovian

        # set learnablility of physics
        # for param in self.mech_ode.r.parameters():
        #     param.requires_grad = learn_physics
        if learn_physics:
            print("Warning: learning the A (interactions) and r parameter in the ODE (growth rate).")
            self.rhs.mech_ode.r.requires_grad = True
            self.rhs.mech_ode.A.requires_grad = True

        # set learnability of scale parameters
        self.Gamma_scale.requires_grad = learn_ObsCov
        self.C_scale.requires_grad = learn_StateCov

    def solve(self, x0, t, controls=None, params={}):
        # solve the ODE using the initial conditions x0 and time points t
        if controls is not None:
            raise NotImplementedError("Controls not implemented yet.")
        x = odeint_wrapper(self.rhs, x0, t, **params)
        return x

    def compute_constant_K(self):
        H = self.h_obs.weight
        Gamma_cov = self.Gamma_scale**2 * self.Gamma_cov.weight
        C_cov = self.C_scale**2 * self.C_cov.weight
        self.S = H @ C_cov @ H.T + Gamma_cov
        self.S_inv = torch.inverse(self.S)
        self.K = C_cov @ H.T @ self.S_inv

    def no_assim(self, y_obs, times):
        """This function is used to simulate the model using y_obs[0] as the initial condition"""

        batch_size = y_obs.shape[0]
        n_times = y_obs.shape[1]

        # Pre-allocation of S_values
        S_values = torch.zeros((batch_size, n_times, self.dim_obs, self.dim_obs)).to(y_obs)
        S_inv_values = torch.zeros((batch_size, n_times, self.dim_obs, self.dim_obs)).to(y_obs)

        S_values[:] = self.S.unsqueeze(0).unsqueeze(0)
        S_inv_values[:] = self.S_inv.unsqueeze(0).unsqueeze(0)

        # x_init: concatenate y_obs[:, 0] with remaining states self.x0[self.dim_obs:]
        x_init = torch.cat((y_obs[:, 0], self.x0[self.dim_obs:].view(1, -1)), dim=1)

        # solve the ODE using the initial conditions x0 and time points t
        x_pred = self.solve(x_init, times, params=self.odeint_params)
        # x_pred: (N_times, N_batch, dim_state)
        # y_obs: (N_batch, N_times, dim_obs)
        # make x_pred: (N_batch, N_times, dim_state)
        x_pred = x_pred.permute(1, 0, 2)
        x_assim = x_pred

        # map to observation space
        y_pred = x_pred[:, :, : self.dim_obs]
        y_assim = x_pred[:, :, : self.dim_obs]

        return (y_pred, y_assim, x_pred, x_assim, S_values, S_inv_values)

    def var3d(self, y_obs, times):
        # y_obs: (N_batch, N_times, dim_obs)
        # times: (N_times)
        batch_size = y_obs.shape[0]
        n_times = y_obs.shape[1]

        # update self.K and self.cov
        self.compute_constant_K()

        # Pre-allocation of S_values
        S_values = torch.zeros((batch_size, n_times, self.dim_obs, self.dim_obs))
        S_inv_values = torch.zeros((batch_size, n_times, self.dim_obs, self.dim_obs))

        S_values[:] = self.S.unsqueeze(0).unsqueeze(0)
        S_inv_values[:] = self.S_inv.unsqueeze(0).unsqueeze(0)

        # Need to make sure these tensors are on correct device.
        # Easiest way was to assign them to same device as y_obs.
        x_assim = (
            torch.zeros((batch_size, y_obs.shape[1], self.dim_state)).to(y_obs).detach()
        )
        x_pred = (
            torch.zeros((batch_size, y_obs.shape[1], self.dim_state)).to(y_obs).detach()
        )
        y_pred = torch.zeros_like(y_obs).to(y_obs)
        y_assim = torch.zeros_like(y_obs).to(y_obs).detach()

        x_pred_n = torch.zeros((batch_size, self.dim_state)).to(y_obs)
        x_pred_n[:] = self.x0
        y_pred_n = self.h_obs(x_pred_n)

        x_pred[:, 0] = x_pred_n.detach().clone()
        y_pred[:, 0] = y_pred_n

        # loop over times
        for n in range(n_times):
            # perform the filtering/assimilation step w/ constant gain K

            x_assim_n = x_pred_n + (self.K @ (y_obs[:, n] - y_pred_n).T).T

            x_assim[:, n] = x_assim_n.detach().clone()
            y_assim[:, n] = self.h_obs(x_assim_n.detach()).detach().clone()

            if n < y_obs.shape[1] - 1:
                # predict the next state by solving the ODE from t_n to t_{n+1}
                x_pred_n = self.solve(
                    x_assim_n, times[n : n + 2], params=self.odeint_params
                )[-1]
                x_pred[:, n + 1] = x_pred_n.detach().clone()

                # compute the observation map
                y_pred_n = self.h_obs(x_pred_n.clone())
                y_pred[:, n + 1] = y_pred_n.clone()

        return (y_pred, y_assim, x_pred, x_assim, S_values, S_inv_values)

    # def compute_covariance(self, ensemble):
    # ensemble is a tensor of shape (N_batch, N_ensemble, dim_state)
    # mean = torch.mean(ensemble, axis=1)
    # deviations = ensemble.permute(0, 2, 1) - mean[..., None]
    # return (1 / (self.N_ensemble - 1)) * deviations.T @ deviations

    def enkf(self, y_obs, times):
        batch_size = y_obs.shape[0]

        # load parameter weights
        H = self.h_obs.weight
        Gamma_cov = self.Gamma_scale**2 * self.Gamma_cov.weight
        C_cov = self.C_scale**2 * self.C_cov.weight

        # Pre-allocation of S_values
        S_values = torch.zeros((batch_size, len(times), self.dim_obs, self.dim_obs))
        S_inv_values = torch.zeros((batch_size, len(times), self.dim_obs, self.dim_obs))

        # Initialization of ensemble members with learnable normal distribution.
        dist_init = torch.distributions.MultivariateNormal(
            self.prior_mean, self.prior_cov.weight
        )
        ensemble = dist_init.sample(
            (
                batch_size,
                self.N_ensemble,
            )
        )

        ensemble_pred = torch.zeros_like(ensemble)
        ensemble_assim = torch.zeros_like(ensemble)

        y_pred_mean = torch.zeros_like(y_obs).to(y_obs)
        y_assim_mean = torch.zeros_like(y_obs).to(y_obs).detach()

        x_pred_mean = torch.zeros((batch_size, y_obs.shape[1], self.dim_state)).detach()
        x_assim_mean = torch.zeros(
            (batch_size, y_obs.shape[1], self.dim_state)
        ).detach()

        # Loop over times
        for n in range(len(times)):
            # Prediction step
            # ensemble_pred = self.solve(ensemble, time)
            ensemble_pred = self.solve(
                ensemble, times[n : n + 2], params=self.odeint_params
            )[-1]

            # add state noise to ensemble pred
            # C_cov: dim_state x dim_state
            # ensemble_pred: (batch_size, N_ensemble, dim_state)
            dist_state = torch.distributions.MultivariateNormal(
                torch.zeros(self.dim_state).to(y_obs), C_cov
            )
            # Sample state noise for each ensemble member
            # The shape of noise will be (batch_size, N_ensemble, dim_state)
            noise = dist_state.sample((batch_size, self.N_ensemble)).view(
                batch_size, self.N_ensemble, self.dim_state
            )

            # Add the sampled noise to ensemble_pred
            ensemble_pred += noise
            # Output shape: (batch_size, N_ensemble, dim_state)

            y_pred_ensemble = self.h_obs(ensemble_pred)

            x_pred_mean[:, n, :] = torch.mean(ensemble_pred, axis=1)
            y_pred_mean[:, n, :] = torch.mean(y_pred_ensemble, axis=1)

            # Compute covariance
            x_cov = batch_covariance(ensemble_pred)
            # Compute S_k
            # S_k = H @ x_cov @ H.T + Gamma_cov
            S_k = torch.matmul(
                torch.matmul(H.unsqueeze(0), x_cov), H.unsqueeze(0).transpose(-1, -2)
            ) + Gamma_cov.unsqueeze(0)
            S_values[:, n] = S_k

            S_k_inv = torch.inverse(S_k)
            S_inv_values[:, n] = S_k_inv

            # Compute Kalman gain
            # K = x_cov @ H.T @ S_k_inv
            K = torch.matmul(
                torch.matmul(x_cov, H.unsqueeze(0).transpose(-1, -2)), S_k_inv
            )
            # Output shape: (batch_size, dim_state, dim_obs)

            # Define the distribution for perturbed observations
            dist_obs = torch.distributions.MultivariateNormal(y_obs[:, n], Gamma_cov)

            # Sample perturbed observations
            perturbed_obs = dist_obs.sample((self.N_ensemble,)).permute(1, 0, 2)
            # Output shape: (batch_size, N_ensemble, dim_obs)

            # Correct way to repeat perturbed_obs for each batch
            # Reshape perturbed_obs to add a batch dimension before repeating
            # perturbed_obs = perturbed_obs.unsqueeze(0).repeat(batch_size, 1, 1)

            # # Perturb observations by zero-mean multivariate normal with covariance Gamma_cov
            # dist_obs = torch.distributions.MultivariateNormal(y_obs[:, n], Gamma_cov)
            # perturbed_obs = dist_obs.sample((self.N_ensemble,)).unsqueeze(0).repeat(batch_size, 1, 1)  # Repeat for each batch
            # perturbed_obs = dist_obs.sample((self.N_ensemble,))

            # Update step
            # Reshape tensors to align for broadcasting in batch operations
            ensemble_pred = ensemble_pred.view(
                batch_size, self.N_ensemble, self.dim_state
            )
            y_pred_ensemble = y_pred_ensemble.view(
                batch_size, self.N_ensemble, self.dim_obs
            )
            perturbations = (perturbed_obs - y_pred_ensemble).transpose(
                1, 2
            )  # Transpose for correct broadcasting

            # Perform batched update
            ensemble_assim = ensemble_pred + torch.matmul(K, perturbations).transpose(
                1, 2
            )
            # Output shape: (batch_size, N_ensemble, dim_state)

            # y_assim_ensemble = self.h_obs(ensemble_assim.view(-1, self.dim_state)).view(
            #     batch_size, self.N_ensemble, self.dim_obs
            # )
            y_assim_ensemble = self.h_obs(ensemble_assim)

            # # Update step
            # ensemble_assim = ensemble_pred + K @ (perturbed_obs - y_pred_ensemble).T
            # y_assim_ensemble = self.h_obs(ensemble_assim)

            x_assim_mean[:, n, :] = torch.mean(ensemble_assim, axis=1)
            y_assim_mean[:, n, :] = torch.mean(y_assim_ensemble, axis=1)

            # Prepare for the next iteration
            ensemble = ensemble_assim.clone()

        return (
            y_pred_mean,
            y_assim_mean,
            x_pred_mean,
            x_assim_mean,
            S_values,
            S_inv_values,
        )

    def forward(self, y_obs, times):
        if self.da_name == "3dvar":
            return self.var3d(y_obs, times)
        elif self.da_name == "enkf":
            return self.enkf(y_obs, times)
        elif self.da_name == "none":
            return self.no_assim(y_obs, times)
        else:
            raise ValueError('da_name must be "3dvar" or "enkf"')

        # # run 3dvar
        # y_pred, y_assim, x_pred, x_assim, cov = self.var3d(y_obs, times)

        # return self.run_da(y_obs, times)


# build a wrapper for FeedForwardNN that allows it to be applied to a batch of inputs of size 1 or act as a fully connected network
class FeedForwardNN_Wrapper(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        control_size=3,
        num_hidden_layers=1,
        layer_width=50,
        activations="gelu",
        dropout=0.01,
        include_control=True,
        fully_connected=False,
        shared_weights=False,
    ):
        super(FeedForwardNN_Wrapper, self).__init__()
        self.fully_connected = fully_connected
        self.shared_weights = shared_weights
        self.include_control = include_control

        # Adjust for control inputs
        if fully_connected:
            inner_input_size = input_size
            inner_output_size = output_size
        else:
            inner_input_size = 1
            inner_output_size = 1

        if include_control:
            inner_input_size += control_size

        # Initialize the network based on the configuration
        if fully_connected or shared_weights:
            self.net = FeedForwardNN(
                inner_input_size,
                inner_output_size,
                num_hidden_layers,
                layer_width,
                activations,
                dropout,
            )
        else:
            total_input_size = input_size
            if include_control:
                total_input_size += control_size

            # create an adjacency matrix for the network
            # it should only allow the i'th input to connect to the i'th output
            # if include control, then the control inputs (which are concatenated to the end of the state vector) SHOULD connect to the output.
            # nout x nin
            adjacency = np.zeros((output_size, total_input_size))
            for i in range(output_size):
                adjacency[i, i] = 1
                if include_control:
                    # include the last control_size columns
                    adjacency[i, -control_size:] = 1

            self.net = StrNN(
                nin=total_input_size,
                nout=output_size,
                hidden_sizes=[layer_width] * num_hidden_layers,
                activation=activations,
                ian_init=False,
                adjacency=adjacency,
            )

    def forward(self, x, u):
        '''

        INPUTS:
           x: (N_batch, input_size)
           u: (N_batch, control_size)

        OUTPUTS:
            output: (N_batch, input_size)
           '''
        if self.fully_connected:
            # Concatenate control inputs if required
            xu = (
                torch.cat((x, u), dim=1)
                if self.include_control and u is not None
                else x
            )
            return self.net(xu)
        else:
            if self.shared_weights:
                u_expanded = u.unsqueeze(1).expand(-1, x.size(1), -1)

                # concatenate control inputs if required
                # should result in shape (N_batch, input_size, control_size + 1)
                xu = (
                    torch.cat((x.unsqueeze(2), u_expanded), dim=2)
                    if self.include_control and u is not None
                    else x.unsqueeze(2)
                )
                return self.net(xu).squeeze(2)
            else:
                xu = (
                    torch.cat((x, u), dim=1)
                    if self.include_control and u is not None
                    else x
                )
                return self.net(xu)

class FeedForwardNN(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        num_hidden_layers=1,
        layer_width=50,
        activations="gelu",
        dropout=0.01,
    ):
        super(FeedForwardNN, self).__init__()

        if not isinstance(layer_width, list):
            layer_width = [layer_width] * (num_hidden_layers)

        if not isinstance(activations, list):
            activations = [activations] * (num_hidden_layers)

        # Ensure the number of widths and activations match the number of hidden layers
        assert len(layer_width) == len(activations) == (num_hidden_layers)

        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, layer_width[0]))
        layers.append(get_activation(activations[0]))
        layers.append(nn.Dropout(p=dropout))  # Dropout layer added here

        # Hidden layers
        for i in range(1, len(layer_width)):
            layers.append(nn.Linear(layer_width[i - 1], layer_width[i]))
            layers.append(get_activation(activations[i]))
            layers.append(nn.Dropout(p=dropout))  # Dropout layer added here

        # Output layer
        layers.append(nn.Linear(layer_width[-1], output_size))

        # Sequentially stack the layers
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class F_Physics(nn.Module):
    def __init__(self, ode=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.ode = ode

    def forward(self, x, t):
        return self.ode.rhs(t, x)


# Define a class for the learned ODE model
# This has a forward method to represent a RHS of an ODE, where rhs = f_physics + f_nn
class HybridODE(nn.Module):

    def __init__(
        self,
        dim_state_mech: int = 10, # mechanistic state dimension
        dim_state_latent: int = 0, # latent state dimension
        normalizer: object = None,
        ode: object = None,
        use_physics: bool = False,
        use_nn_markovian: bool = True,
        use_nn_non_markovian: bool = False,
        num_hidden_layers=1,
        layer_width=50,
        activations="gelu",
        dropout=0.01,
        low_bound=1e5,
        high_bound=1e12,
        low_bound_latent=0,
        high_bound_latent=1,
        nn_coefficient_scaling=1e3,
        pre_multiply_x: bool = True,
        include_control: bool = True,
        fully_connected: bool = True,
        shared_weights: bool = False,
    ):
        super(HybridODE, self).__init__()
        self.use_physics = use_physics
        self.use_nn_markovian = use_nn_markovian
        self.use_nn_non_markovian = use_nn_non_markovian
        self.normalizer = normalizer

        # These bounds are used to clamp the state variables AND then the same values are used to clamp the overall RHS.
        self.low_bound = low_bound
        self.high_bound = high_bound
        self.low_bound_latent = low_bound_latent
        self.high_bound_latent = high_bound_latent

        self.nn_coefficient_scaling = nn_coefficient_scaling
        self.pre_multiply_x = pre_multiply_x

        self.dim_state_mech = dim_state_mech
        self.dim_state_latent = dim_state_latent
        # nn_coefficient_scaling is a scaling factor for the output of the neural network
        # it is only active if pre_multiply_x is False

        self.mech_ode = ode

        # always build the NNs, but only use them if use_nn is True

        # building 2 NNs: one for the mechanistic state and one for the latent state
        # each NN will have the same structure, but different inputs

        print(
            "Warning: hard-coded 3 control categories (binary) to be appended to NN state input. Also assuming synchronous control times across subjects."
        )
        self.f_nn_markovian = FeedForwardNN_Wrapper(
            dim_state_mech, # mechanistic state dimension
            dim_state_mech, # mechanistic state dimension
            3,  # hard-coding control-size of 3
            num_hidden_layers,
            layer_width,
            activations,
            dropout,
            include_control,
            fully_connected,
            shared_weights,
        )

        self.f_nn_non_markovian = FeedForwardNN_Wrapper(
            dim_state_mech+dim_state_latent, # concatenation of mechanistic and latent states
            dim_state_mech+dim_state_latent, # concatenation of mechanistic and latent states
            3,  # hard-coding control-size of 3
            num_hidden_layers,
            layer_width,
            activations,
            dropout,
            include_control,
            fully_connected,
            shared_weights,
        )

    def forward(self, t, x):
        D = self.dim_state_mech
        L = self.dim_state_latent

        # print(f"x({t}): ", x)
        if torch.any(torch.isnan(x)):
            bp()

        # clamp x to be in range [1e5, 1e12] ...recall that 1e5 is detection threshold
        # clamp x in up to D dimensions
        # in latent dimensions clamp [-1, 1]
        x = torch.cat(
            (
                torch.clamp(
                    x[:, :D], self.low_bound, self.high_bound
                ),  # mechanistic state clamp
                torch.clamp(
                    x[:, D:], self.low_bound_latent, self.high_bound_latent
                ),  # latent state clamp
            ),
            dim=1,
        )

        if self.use_physics:
            # errored due to in-place operation
            # rhs[:, :D] = rhs[:, :D] + self.mech_ode.rhs(t, x)  # self.f_physics(x, t)
            mech_rhs = torch.cat(
                (
                    self.mech_ode.rhs(t, x[:, :D]),
                    torch.zeros_like(x[:, D:], requires_grad=True).to(x),
                ),
                dim=1,
            )
        else:
            mech_rhs = torch.zeros_like(x, requires_grad=True).to(x)

        # determine control categories
        names = ["High Fat Diet", "Vancomycin", "Gentamicin"]
        one_hot_control = torch.zeros(x.shape[0], len(names)).to(x)
        for key in self.mech_ode.pert_params.keys():
            if (
                t >= self.mech_ode.times[key][0].squeeze()
                and t <= self.mech_ode.times[key][1].squeeze()
            ):
                one_hot_control[:, names.index(key)] = 1

        x_scaled = x.clone()
        # apply normalization to observed components of x
        x_scaled[:, :D] = self.normalizer.encode(x[:, :D])

        # concatenate one-hot control to x_scaled and pass through NN
        # x_scaled: (N_batch, dim_state)
        # one_hot_control: (dim_control)

        if self.use_nn_markovian:
            # this is dimension (N_batch, dim_state_mech), so it will be added to rhs[:, :D]
            nn_raw_markovian = self.f_nn_markovian(x_scaled[:, :D], one_hot_control)

            # sign(nn) * (10^|nn| - 1)
            nn_scaled_markovian = torch.sign(nn_raw_markovian) * (10 ** torch.abs(nn_raw_markovian) - 1)
            # this asks nn output to be in range -10 to 10 or so. May want to have additional derivative scaling so NN outputs ~ [-1, 1].

            # add_term = torch.zeros_like(x, requires_grad=True).to(x)
            # in the observed components of x, do x*NN, else do NN
            if self.pre_multiply_x:
                foo = x[:, :D] * nn_scaled_markovian
            else:
                foo = self.nn_coefficient_scaling * nn_scaled_markovian

            add_markovian = torch.cat(
                        (foo,
                        torch.zeros_like(x[:, D:], requires_grad=True).to(x),
                        ),
                        dim=1,
                    )
        else:
            add_markovian = torch.zeros_like(x, requires_grad=True).to(x)

        if self.use_nn_non_markovian:
            nn_raw_non_markovian = self.f_nn_non_markovian(x_scaled, one_hot_control)

            # do re-scalings of nn_raw_non_markovian in all dimensions (including latent)
            nn_scaled_non_markovian = torch.sign(nn_raw_non_markovian) * (10 ** torch.abs(nn_raw_non_markovian) - 1)

            if self.pre_multiply_x:
                add_non_markovian = x * nn_scaled_non_markovian
            else:
                # Note that this will work poorly if hidden dims are constrained to be small!
                add_non_markovian = self.nn_coefficient_scaling * nn_scaled_non_markovian
        else:
            add_non_markovian = torch.zeros_like(x, requires_grad=True).to(x)

        # add all terms together
        rhs = mech_rhs + add_markovian + add_non_markovian

        if torch.any(torch.isnan(rhs)):
            # print the index where the nan is
            bp()

        # clamp rhs to be non-negative for states where x is at the lower bound
        # and clamp rhs to be non-positive for states where x is at the upper bound
        # only apply this to the observed components of x

        # mech
        min_condition = x[:, :D] == self.low_bound
        rhs[:, :D][min_condition] = torch.clamp_min(rhs[:, :D][min_condition], 0)

        # latent
        min_condition = x[:, D:] == self.low_bound_latent
        rhs[:, D:][min_condition] = torch.clamp_min(rhs[:, D:][min_condition], 0)

        # mech
        max_condition = x[:, :D] == self.high_bound
        rhs[:, :D][max_condition] = torch.clamp_max(rhs[:, :D][max_condition], 0)

        # latent
        max_condition = x[:, D:] == self.high_bound_latent
        rhs[:, D:][max_condition] = torch.clamp_max(rhs[:, D:][max_condition], 0)


        # finally, clamp rhs to be in range of [low_bound, high_bound] for observed components of x and [low_bound_latent, high_bound_latent] for latent components of x
        rhs = torch.cat(
            (
                torch.clamp(
                    rhs[:, :D], self.low_bound, self.high_bound
                ),  # mechanistic state clamp
                torch.clamp(
                    rhs[:, D:], self.low_bound_latent, self.high_bound_latent
                ),  # latent state clamp
            ),
            dim=1,
        )

        # print(f"rhs({t}): ", rhs)
        return rhs
