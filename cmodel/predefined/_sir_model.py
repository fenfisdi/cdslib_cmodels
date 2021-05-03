from typing import List

from cmodel.model import CompartmentalModel


class ModelSIR(CompartmentalModel):

    def build_model(
        self, t, y,
        Lmbd, mu, omega, gamma, chi, eta, Pi, tau
    ) -> List[float]:
        """
        Returns the vector field dy/dt evaluated at a given point in phase space
        """
        S, I, R = y

        dydt = [
            Lmbd + Pi * R - tau * S * I - (chi + mu) * S,
            tau * S * I - eta * R - (gamma + omega + mu) * I,
            gamma * I + chi * S - (eta + Pi + mu) * R
        ]

        return dydt
