import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
from tqdm import tqdm
from pennylane_subroutines import basic_simulation, evaluate_commutators, fermion_chain_1d, load_hamiltonian, split_hamiltonian

import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')

modes = 7
coupling = 1
time_steps = 1/(2**np.array(range(1, 8)))
n_steps = 1
method = 'Commutator'
random_weights = True
device = 'default.qubit'

#H = load_hamiltonian()
#H0, H1 = split_hamiltonian(H) #todo: change bakc
H0 = qml.Hamiltonian(coeffs = [1j], observables=[qml.PauliX(0)])
H1 = qml.Hamiltonian(coeffs = [1j], observables=[qml.PauliZ(0)])
hamiltonian = (H0, H1, coupling)
n_wires = 1


# A manual alternative
#H0, H1 = fermion_chain_1d(modes, random_weights=random_weights)
#evaluate_commutators(H0, H1)
#hamiltonian = (qml.jordan_wigner(H0), qml.jordan_wigner(H1), coupling)
#n_wires = modes

dev = qml.device(device, wires=n_wires)

method_errors={}
for commutator_method in tqdm(['NCP_3_6', 'NCP_4_10', 'PCP_5_16', 'PCP_6_26', 'PCP_4_12', 'NCP_5_18'], desc='Methods'):
    method_errors[commutator_method] = []
    for time in time_steps:
        error = basic_simulation(hamiltonian, time, n_steps, dev, n_wires, 
                                n_samples = 3, method = method, commutator_method = commutator_method,
                                approximate = False)
        method_errors[commutator_method].append(error)

# Plot the results
ax, fig = plt.subplots()
for method in method_errors.keys():
    plt.plot(time_steps, method_errors[method], '-o')

plt.yscale('log')
plt.xscale('log')

plt.xlabel('Time step')
plt.ylabel('Error')

plt.legend(method_errors.keys())

plt.savefig('method_errors.pdf', bbox_inches='tight', format='pdf')