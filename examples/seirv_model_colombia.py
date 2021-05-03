import sys
import os
from time import time
from typing import Any, Dict, List, Union

import matplotlib.pyplot as plt
import pandas as pd

# Add project root and this file's directories to path in order to find cmodel
# package
this_file_dir = os.path.dirname(__file__)
project_root_dir = os.path.join(this_file_dir, '..')
sys.path.append(this_file_dir)
sys.path.append(project_root_dir)

from cmodel import model, predefined
# State variables and parameters are setup in col_vars_params.py module
from examples.col_vars_params import (
    seirv_state_variables_colombia,
    seirv_parameters_colombia
)


# Time span and time steps in days
t_span_col = [0, 171]
t_steps_col = 172

infected_reference_col_path = os.path.join(
    this_file_dir, '..', 'example_data', 'infected_reference_col.csv'
)

infected_reference_col = pd.read_csv(
    infected_reference_col_path
)['infected_reference_col'].to_numpy().T


def seirv_model_example(
    state_variables: List[model.StateVariable] = seirv_state_variables_colombia,
    parameters: List[model.Parameter] = seirv_parameters_colombia,
    t_span: List[Union[float, int]] = t_span_col,
    t_steps: int = t_steps_col,
    infected_reference: List[int] = infected_reference_col,
    monitor_computation_time: bool = False,
    plot_solution: bool = False,
) -> Dict[str, Any]:

    # Instantiate the model
    model_SEIRV = predefined.ModelSEIRV(
        state_variables=state_variables,
        parameters=parameters,
        t_span=t_span,
        t_steps=t_steps
    )

    # In the next iteration of cmodel's API the user will need to define the
    # model here manually by specifying the fluxes of each state variable using
    # some interface such as model_SEIRV.add_flux(...)

    # Run the model
    t0 = time()
    solution = model_SEIRV.run_model(method='RK45')
    if monitor_computation_time:
        print(f"Time calculating solution: {time() - t0}")

    if plot_solution:
        plt.figure()
        plt.plot(solution.t, solution.y[2], "ko-", label='SEIRV model')
        plt.plot(solution.t, infected_reference, "ro-", label='Data from INS')
        plt.xlabel('Days since case 1')
        plt.ylabel('Confirmed infected people')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
        plt.close()

    seirv_model_example = {
        'model': model_SEIRV,
        'solution': solution
    }

    return seirv_model_example


if __name__ == "__main__":
    seirv_model_example(
        monitor_computation_time=True,
        plot_solution=True
    )
