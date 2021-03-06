from typing import List

import pytest
from pytest import approx
import numpy as np

from cmodel import model, predefined


@pytest.fixture
def state_variables_source():
    state_variables_source = [
        ("suceptible", "S", 50372424),
        ("exposed", "Ex", 0),
        ("infected", "If", 1),
        ("recovered", "R", 0),
        ("vaccinated", "V", 0),
    ]

    return state_variables_source


@pytest.fixture
def state_variables(state_variables_source):
    state_variables = [
        model.StateVariable(*sv) for sv in state_variables_source
    ]

    return state_variables


@pytest.fixture
def parameters_source():
    parameters_source = [
        ("Lambda", "Lambda", 2083.62),
        ("mu", "mu", 0.0000150767),
        ("alpha", "alpha", 0.172414),
        ("omega", "omega", 0.0114),
        ("gamma", "gamma", 0.0666667),
        ("xi1", "xi1", 1.09736),
        ("xi2", "xi2", 6.32609),
        ("sigma", "sigma", 1.),
        ("b1", "b1", 2.91823E-9),
        ("b2", "b2", 2.67472E-9),
        ("b3", "b3", 4.635E-9),
        ("c1", "c1", 0.0000855641),
        ("c2", "c2", 2.94906E-7),
        ("c3", "c3", 0.159107),
    ]
    return parameters_source


@pytest.fixture
def parameters(parameters_source):
    """The values of the parameters were obtained from Boris' optimization
    script.
    """
    parameters = [model.Parameter(*param) for param in parameters_source]

    return parameters


@pytest.fixture
def model_SEIRV(state_variables, parameters):
    t_span = [0, 171]
    t_steps = 172

    model_SEIRV = predefined.ModelSEIRV(state_variables, parameters, t_span, t_steps)

    return model_SEIRV


def test_state_variables_init_vals(
    model_SEIRV: predefined.ModelSEIRV,
    state_variables_source
):
    sv_init_vals = [sv[2] for sv in state_variables_source]
    assert model_SEIRV.state_variables_init_vals == sv_init_vals


def test_parameters_init_vals(model_SEIRV: predefined.ModelSEIRV, parameters_source):
    params = [param[2] for param in parameters_source]
    assert model_SEIRV.parameters_init_vals == params


@pytest.mark.parametrize(
    'bounds',
    [
        [1, 2],
        [1., 2],
        [1, 2.],
        [1., 2.]
    ]
)
def test_parameters_bounds(bounds):
    try:
        model.Parameter("a", "b", 1.5, bounds=bounds)
    except AttributeError:
        assert False
    else:
        assert True


@pytest.mark.parametrize(
    'initial_value,bounds',
    [
        (1.3, "a"),
        (1.5, (1, 2.)),
        (1.5, [1, 2, 3]),
        (1.5, ["a", 2.]),
        (1.5, [1, "b"]),
        (1.5, [2, 1]),
        (0, [1, 2]),
    ]
)
def test_parameters_bounds_error_handling(initial_value, bounds):
    with pytest.raises(AttributeError):
        model.Parameter("a", "b", initial_value, bounds=bounds)


def test_state_variables_initialization(
    state_variables: List[model.StateVariable], state_variables_source
):
    state_variables_source_test = [
        (sv.name, sv.representation, sv.initial_value) for sv in state_variables
    ]

    assert state_variables_source_test == state_variables_source


def test_parameters_initialization(
    parameters: List[model.Parameter], parameters_source
):
    parameters_source_test = [
        (param.name, param.representation, param.initial_value) for param in parameters
    ]

    assert parameters_source_test == parameters_source


def test_model_SEIRV_build_model(
    model_SEIRV: predefined.ModelSEIRV,
):
    """Tests if evaluation of differential equation defined in SEIRV model is OK"""

    state_variables = model_SEIRV.state_variables_init_vals
    parameters = model_SEIRV.parameters_init_vals

    # Expected value calculated in Mathematica, in Boris' script
    expected_diff_eqn_value = approx(
        [1324.04, 0.134732, -0.0780818, 0.0666667, 6.32609], rel=0.99
    )

    # Evaluation of differential equation must be independent of time
    for t in range(10):
        diff_eqn_value = model_SEIRV.build_model(
            t, state_variables, *parameters
        )
        assert diff_eqn_value == expected_diff_eqn_value


def test_model_SEIRV_run_model_initial_value(
    model_SEIRV: predefined.ModelSEIRV,
    state_variables: List[model.StateVariable]
):
    initial_state_variables = model_SEIRV.state_variables_init_vals
    param_types = (list, tuple, np.array)
    parameters = model_SEIRV.parameters_init_vals
    for param_type in param_types:
        solution = model_SEIRV.run_model(parameters=param_type(parameters))
        solution_initial_values = [y[0] for y in solution.y]
        assert solution_initial_values == approx(initial_state_variables)
