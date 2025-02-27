import torch
from torch import nn
from torch.distributions import MultivariateNormal
from rl_policy_gradient.models.twoheadedmlp import TwoHeadedMLP


class StochasticPolicyNetwork(TwoHeadedMLP):
    """
    A multilayer perceptron with two output heads to represent a multivariate Gaussian distribution.
    This type of neural network can be used to output a (multivariate) Gaussian distribution,
    in this case one output head is for the mean and the other are elements of the covariance matrix.

    NOTE: In the 1D case reduces to a mean and std output.
    """

    def __init__(self, input_size: int, output_size: int):
        """
        Initialize the MLP with two output heads.

        :param input_size: number of input features
        :param output_size: number of output features
        """
        super(StochasticPolicyNetwork, self).__init__()
        self.layers = nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU()
        )
        #linear layer for mean
        self.mean_head = torch.nn.Linear(32, output_size)
        #https://en.wikipedia.org/wiki/Cholesky_decomposition
        #linear layer for log diagonal of Cholesky decomposition
        #the log is for numerical stability.
        self.log_diag_chol_head = torch.nn.Linear(32, output_size)
        self.double()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the MLP.
        :param x: input tensor of shape (batch_size, input_size)
        :return: tuple containing mean tensor of shape (batch_size, output_size)
                 and log diagonal of Cholesky decomposition tensor of shape (batch_size, output_size)
        """
        features = self.layers(x)
        mean = self.mean_head(features)
        log_diag_chol = self.log_diag_chol_head(features)
        return mean, log_diag_chol

    def predict(self, x: torch.Tensor) -> MultivariateNormal:
        """
        Predict a MultivariateNormal distribution given an input tensor.
        :param x: input tensor of shape (batch_size, input_size)
        :return: multivariateNormal distribution
        The forward pass gives a mean and log of the diagonal elements of the Cholesky decomposition
        of the covariance matrix. For a full prediction, we need to construct the Multivariate
        Gaussian distribution, and for that, we need to exponentiate the log of the diagonal elements
        and then reconstruct the covariance matrix using the Cholesky decomposition.
        (https://en.wikipedia.org/wiki/Covariance_matrix)
        """
        mean, log_diag_chol = self(x)
        #construct the covariance matrix using Cholesky decomposition
        cov_matrix = torch.diag_embed(torch.exp(log_diag_chol))
        return MultivariateNormal(mean, cov_matrix)

    def sample_action(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action from a Gaussian policy modeled by the provided model and compute the log probability of the action.
        :param state: the input state tensor
        :return: a tuple containing the sampled action tensor and the log probability of the action
        """
        action_distribution = self.predict(state)
        action: torch.Tensor = action_distribution.sample()
        ln_prob: torch.Tensor = action_distribution.log_prob(action)
        return action, ln_prob

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the log probability of an action given a state and a model representing the policy.
        :param state: the input state tensor
        :param action: the action tensor
        :return: the log probability of the action given the state and policy model
        """
        distribution = self.predict(state)
        return distribution.log_prob(action)