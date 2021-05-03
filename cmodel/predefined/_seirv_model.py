from typing import List

from cmodel.model import CompartmentalModel


class ModelSEIRV(CompartmentalModel):

    def build_model(
        self, t, y,
        Lmbd, mu, inv_alpha, omega, gamma, xi_E, xi_I, sigma, beta_E, beta_I, beta_V, c_E, c_I, c_V
    ) -> List[float]:
        """
        Returns the vector field dy/dt evaluated at a given point in phase space
        """
        S, E, I, R, V = y

        def beta(x, b, c):
            return b / (1. + c * x)

        principal_flux = S * (beta(E, beta_E, c_E) * E + beta(I, beta_I, c_I) * I + beta(V, beta_V, c_V) * V)

        dydt = [
            Lmbd - principal_flux - S * mu,
            principal_flux - (inv_alpha + mu) * E,
            inv_alpha * E - (omega + gamma + mu) * I,
            gamma * I - mu * R,
            xi_E * E + xi_I * I - sigma * V
        ]

        return dydt
