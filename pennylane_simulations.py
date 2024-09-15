import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
from tqdm import tqdm
from pennylane_subroutines import time_simulation, load_hamiltonian, split_hamiltonian

import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')

time_steps = 1/(2**np.array(range(1, 8)))
n_steps = 1
method = 'Commutator'
random_weights = True
device = 'default.qubit'
commutator_method = 'PCP_6_26'

H, n_wires = load_hamiltonian(layout="1x8")
H0, H1 = split_hamiltonian(H)
hamiltonian = (H0, H1)

dev = qml.device(device, wires=n_wires)

method_errors={}
for method in tqdm(['SymZassenhaus4', 'Suzuki4', 'Yoshida4'], desc='Methods'):
    method_errors[method] = []
    for time in time_steps:
        error = time_simulation(hamiltonian, time, n_steps, dev, n_wires, 
                                n_samples = 3, method = method, commutator_method = commutator_method,
                                approximate = True)
        method_errors[method].append(error)

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