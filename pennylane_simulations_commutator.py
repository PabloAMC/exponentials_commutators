import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
from tqdm import tqdm
from pennylane_subroutines_commutator import time_simulation, evaluate_commutators, fermion_chain_1d, load_hamiltonian, split_hamiltonian

import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')

coupling = 1
time_steps = 1/(2**np.array(range(1, 8)))
n_steps = 1
method = 'Commutator'
random_weights = True
device = 'default.qubit'
layout = "1x8"

if layout == "1x1":
    H0 = qml.Hamiltonian(coeffs = [1., 0.], observables = [qml.PauliX(0), qml.Identity(0)])
    H1 = qml.Hamiltonian(coeffs = [1., 0.], observables = [qml.PauliZ(0), qml.Identity(0)])
    n_wires = 1
    hamiltonian = (H0, H1, 1.)
else:
    H, n_wires = load_hamiltonian(layout=layout)
    H0, H1 = split_hamiltonian(H)
    hamiltonian = (H0, H1, coupling)


dev = qml.device(device, wires=n_wires)

method_errors={}
method_resources = {}
for commutator_method in tqdm(['NCP_3_6', 'NCP_4_10', 'PCP_5_16', 'PCP_6_26', 'PCP_4_12', 'NCP_5_18'], desc='Methods'):
    method_errors[commutator_method] = []
    method_resources[commutator_method] = []
    for time in time_steps:
        error, resources = time_simulation(hamiltonian, time, n_steps, dev, n_wires, 
                                n_samples = 10, method = method, commutator_method = commutator_method,
                                approximate = True)
        method_errors[commutator_method].append(error)
        res = (resources.gate_types['RX'] + resources.gate_types['RZ'] + resources.gate_types['RY'])/time**2
        method_resources[commutator_method].append(res)

# Plot the results
ax, fig = plt.subplots()
for method in method_errors.keys():
    text, supr, sub = method.split('_')
    plt.plot(time_steps, method_errors[method], label = f"{text}$_{{{sub}}}^{{[{supr}]}}$")

plt.yscale('log')
plt.xscale('log')

plt.xlabel('Time step')
plt.ylabel('Error')

plt.legend(fontsize='small')

plt.savefig(f'results/commutator_errors_{layout}.pdf', bbox_inches='tight', format='pdf')

# Plot the resources
ax, fig = plt.subplots()
for method in method_resources.keys():
    plt.plot(method_resources[method], method_errors[method], label = method)

plt.yscale('log')
plt.xscale('log')

plt.xlabel('1 qubit rotation gates')
plt.ylabel('Error')

plt.legend(fontsize='small')

plt.savefig(f'results/commutator_resources_{layout}.pdf', bbox_inches='tight', format='pdf')