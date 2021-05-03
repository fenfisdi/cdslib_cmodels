from cmodel.model import Parameter, StateVariable
from examples.cmodel_examples_utilities import param_sample_range

# Instantiation of state variables and parameters

# In the next implementation (in which the model is built from scratch)
# StateVariable and Parameter instances initialization ought be a bit
# different, but the idea is the same

# The values of the parameters were obtained from Boris' Mathematica script
# (name, representation, initial_value)
seirv_state_variables_colombia_src = [
    ("suceptible", "S", 50372424),
    ("exposed", "Ex", 0),
    ("infected", "If", 1),
    ("recovered", "R", 0),
    ("vaccinated", "V", 0),
]

# List of instances of state variables
seirv_state_variables_colombia = [
    StateVariable(*sv) for sv in seirv_state_variables_colombia_src
]

# Parameters not variated by Boris' Mathematica script
Lmbd = 2083.62
mu = 0.0000150767
inv_alpha = 0.172414
omega = 0.0114
gamma = 0.0666667
xi_I = 6.32609
sigma = 1.0

# Optimal parameters found by Boris' Mathematica script
xi_E_opt = 1.09736
b_E_opt = 2.91823E-9
b_I_opt = 2.67472E-9
b_V_opt = 4.635E-9
c_E_opt = 0.0000855641
c_I_opt = 2.94906E-7
c_V_opt = 0.159107

# Parameters used to define their own bounds via the function
# param_sample_range
xi_E = 2.3
b_E, b_I, b_V = 1.5 * 3.11e-9, 1.5 * 0.62e-9, 1.5 * 1.03e-9

# Bounds of c and xi_I parameters used in Boris' Mathematica script
c_bounds = [0, 1]
xi_I_bounds = [0, 10]

# param_range_width used in Boris' Mathematica script
param_range_width = 3

# Each element of the list refers to a parameter
# (name, representation, initial_value, bounds)
seirv_parameters_colombia_src_ini = [
    ("Lambda", "Lmbd", Lmbd, [Lmbd, Lmbd]),
    ("mu", "mu", mu, [mu, mu]),
    ("inv_alpha", "inv_alpha", inv_alpha, [inv_alpha, inv_alpha]),
    ("omega", "omega", omega, [omega, omega]),
    ("gamma", "gamma", gamma, [gamma, gamma]),
    ("xi_E", "xi_E", xi_E_opt, param_sample_range(xi_E, param_range_width)),
    ("xi_I", "xi_I", xi_I, xi_I_bounds),
    ("sigma", "sigma", sigma, [sigma, sigma]),
    ("b_E", "b_E", b_E_opt, param_sample_range(b_E, param_range_width)),
    ("b_I", "b_I", b_I_opt, param_sample_range(b_I, param_range_width)),
    ("b_V", "b_V", b_V_opt, param_sample_range(b_V, param_range_width)),
    ("c_E", "c_E", c_E_opt, c_bounds),
    ("c_I", "c_I", c_I_opt, c_bounds),
    ("c_V", "c_V", c_V_opt, c_bounds),
]
seirv_parameters_colombia_src = [
    {
        'name': param_info[0],
        'representation': param_info[1],
        'initial_value': param_info[2],
        'bounds': param_info[3]
    }
    for param_info in seirv_parameters_colombia_src_ini
]

# List of instances of parameters
seirv_parameters_colombia = []
for param_info in seirv_parameters_colombia_src:
    seirv_parameters_colombia.append(
        Parameter(**param_info)
    )
