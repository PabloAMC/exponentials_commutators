import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
from tqdm import tqdm
from pennylane_subroutines import time_simulation, load_hamiltonian, split_hamiltonian

import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')

time_steps = 1/(2**np.array(range(3, 8)))
n_steps = 1
method = 'Commutator'
random_weights = True
device = 'default.qubit'
commutator_method = 'PCP_6_26'

#H, n_wires = load_hamiltonian(layout="1x4")
#H0, H1 = split_hamiltonian(H)
H0 = qml.Hamiltonian(coeffs = [1., 0.], observables = [qml.PauliX(0), qml.Identity(0)])
H1 = qml.Hamiltonian(coeffs = [1., 0.], observables = [qml.PauliZ(0), qml.Identity(0)])
n_wires = 1
hamiltonian = (H0, H1)

dev = qml.device(device, wires=n_wires)

method_errors={}
method_resources = {}
for method in tqdm(['SymZassenhaus4', 'Suzuki4', 'Yoshida4'], desc='Methods'): # 'NestedCommutator']) 
    method_errors[method] = []
    method_resources[method] = []
    for time in time_steps:
        error, resources = time_simulation(hamiltonian, time, n_steps, dev, n_wires, 
                                n_samples = 20, method = method, commutator_method = commutator_method,
                                approximate = True)
        method_errors[method].append(error)
        res = (resources.gate_types['RX'] + resources.gate_types['RZ'] + resources.gate_types['RY'])/time
        method_resources[method].append(res)

    # Fit exponent to the error method_errors[method] = a * time_steps**b
    b, a = np.polyfit(np.log(time_steps), np.log(method_errors[method]), 1)
    print(f"Method: {method}, Exponent: {b}")

# Plot the results
ax, fig = plt.subplots()
for method in method_errors.keys():
    plt.plot(time_steps, method_errors[method], '-o', label = method)

plt.yscale('log')
plt.xscale('log')

plt.xlabel('Time step')
plt.ylabel('Error')

plt.legend(fontsize='small')

plt.savefig('simulation_errors.pdf', bbox_inches='tight', format='pdf')

# Plot the resources
ax, fig = plt.subplots()
for method in method_resources.keys():
    plt.plot(method_resources[method], method_errors[method], '-o', label = method)

plt.yscale('log')
plt.xscale('log')

plt.xlabel('1 qubit rotation gates')
plt.ylabel('Error')

plt.legend(fontsize='small')

plt.savefig('simulation_resources.pdf', bbox_inches='tight', format='pdf')