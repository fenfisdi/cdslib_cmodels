from typing import List

from cmodel.model import CompartmentalModel


class ModelSEIR(CompartmentalModel):

    def build_model(
        self, t, y,
        Lmbd, mu, omega, gamma, inv_alpha, chi, beta_E, beta_I
    ) -> List[float]:
        """
        Returns the vector field dy/dt evaluated at a given point in phase space
        """
        S, E, I, R = y

        dydt = [
            Lmbd - beta_E * S * E - beta_I * S * I - (chi + mu) * S,
            beta_E * S * E + beta_I * S * I - inv_alpha * E - mu * E,
            inv_alpha * E - (gamma + omega + mu) * I,
            gamma * I + chi * S - mu * R
        ]

        return dydt
