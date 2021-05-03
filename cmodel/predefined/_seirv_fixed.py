from typing import List

from cmodel.model import CompartmentalModel


class ModelSimpleSEIRV(CompartmentalModel):

    def build_model(
        self, t, y,
        Lmbd, mu, omega, gamma, inv_alpha, xi_E, xi_I, sigma, beta_E, beta_I, beta_V
    ) -> List[float]:
        """
        Returns the vector field dy/dt evaluated at a given point in phase space
        """
        S, E, I, R, V = y

        dydt = [
            Lmbd - beta_E * S * E - beta_I * S * I - beta_V * S * V - mu * S,
            beta_E * S * E + beta_I * S * I + beta_V * S * V - inv_alpha * E - mu * E,
            inv_alpha * E - (gamma + omega + mu) * I,
            gamma * I - mu * R,
            xi_E * E + xi_I * I - sigma * V
        ]

        return dydt
