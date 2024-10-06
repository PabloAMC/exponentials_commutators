from functools import partial
import pennylane as qml
import numpy as np
from tqdm import tqdm
from scipy.linalg import logm

import copy
gs = copy.copy(qml.resource.resource.StandardGateSet)
gs.add('StatePrep')

from coefficients.generate_coefficients import NCP_3_6, NCP_4_10, NCP_5_18, PCP_5_16, PCP_6_26

# You can choose from BoseHubbard, FermiHubbard, Heisenberg, Ising, check https://pennylane.ai/datasets/
def load_hamiltonian(name = "Ising", periodicity="open", lattice="chain", layout="1x4"):

    hamiltonians = qml.data.load("qspin", sysname=name, periodicity=periodicity, 
                                lattice=lattice, layout=layout, attributes=["hamiltonians"])

    return hamiltonians[0].hamiltonians[50], eval(layout.replace('x', '*'))

def split_hamiltonian(H):
    H0 = 0*qml.Identity(0)
    H1 = 0*qml.Identity(0)

    for coeff, op in zip(H.coeffs, H.ops):
        if 'X' in str(op.pauli_rep):
            H0 = H0 + coeff * op
        else:
            H1 = H1 + coeff * op
    return H0, H1


def fermion_chain_1d(n, random_weights = False):
    r"""
    Implements a 1D Fermi-Hubbard model with the splitting.

    Arguments:
    ---------
    n: int
        The number of sites
    random_weights: bool
        Whether to use random weights for the Hamiltonian 
    """
    def weights(random):
        if random: return np.random.uniform(-1, 1)
        else: return 1.

    H0 = weights(random_weights) * (qml.FermiC(0) * qml.FermiA(1) + qml.FermiC(1) * qml.FermiA(0))
    H1 = weights(random_weights) * (qml.FermiC(1) * qml.FermiA(2) + qml.FermiC(2) * qml.FermiA(1))

    for i in range(2, n-1, 2):
        H0 += weights(random_weights) * (qml.FermiC(i) * qml.FermiA(i+1) + qml.FermiC(i+1) * qml.FermiA(i))
        H1 += weights(random_weights) * (qml.FermiC((i+1)%n) * qml.FermiA((i+2)%n) + qml.FermiC((i+2)%n) * qml.FermiA((i+1)%n))

    return H0, H1


def LieTrotter_ff(H, h): 
    #qml.exp(H, h)
    qml.TrotterProduct(H, h)


def CommutatorEvolution(
        simulateA, h_power_A,
        simulateB, h_power_B,
        h, cs, positive = True):
    r"""
    Implements the evolution of the commutator term in the LieTrotter step.

    Arguments:
    ---------
    simulateA: Callable(h)
        The first Hamiltonian to simulate
    simulateB: Callable(h)
        The second Hamiltonian to simulate
    h: float
        The time step
    c: float
        The coupling strength of the interaction term
    cs: np.array
        The coefficients of the commutator
    positive: bool
        Whether to use positive or negative counte-palyndromic coefficients
    """
    m = len(cs)-1

    cs_ = cs * (-1)**(int(positive) + 1)
    for c0, c1 in zip(cs_[::2], cs_[1::2]):
        simulateA(h = c0*h)
        simulateB(h = c1*h)
    if m%2 == 0: 
        simulateA(h = cs_[-1]*h)

    if m%2 == 0: 
        simulateA, simulateB = simulateB, simulateA

    cs_ = cs[::-1]
    for c0, c1 in zip(cs_[::2], cs_[1::2]):
        simulateA(h = c0*h)
        simulateB(h = c1*h)
    if m%2 == 0: 
        simulateA(h = cs_[-1]*h)

def DoubleCommutatorEvolution(
        simulateA, simulateB, h, cs
):
    r"""
    Simulates [A, [A, B]] with O(h^4) precision.

    Arguments:
    ---------
    simulateA: Callable(h)
        The first Hamiltonian to simulate
    simulateB: Callable(h)
        The second Hamiltonian to simulate
    h: float
        The time step
    cs: np.array
        The coefficients of the nested commutator.
    """
    for cB, cA in zip(cs[::2], cs[1::2]):
        simulateB(h = cB*h)
        simulateA(h = cA*h)
    if len(cs)%2 != 0:
        simulateB(h = cs[-1]*h)


def LieTrotter(H0, H1, h, cs, positive, reversed = False, 
               commutator = False):
    r"""
    Simulates Lie Trotter step for the Hamiltonian H = hH0 + hH1 -i h*c[H0, H1]

    Arguments:
    ---------
    H0: FermiSentence
        The first (fast-forwardable) Hamiltonian
    H1: FermiSentence
        The second (fast-forwardable) Hamiltonian
    h: float
        The time step
    c: float
        The coupling strength of the interaction term
    commutator_method: str
        The method to compute the commutator.
    reversed: bool
        Whether to reverse the order of the terms in the Hamiltonian

    Returns:
    --------
    None
    """
    simulateA = partial(LieTrotter_ff, H=H0)
    simulateB = partial(LieTrotter_ff, H=H1)

    if not reversed:
        simulateA(h = h)
        simulateB(h = h)
    else:
        simulateB(h = h)
        simulateA(h = h)

def Strang(H0, H1, h, cs, positive):
    LieTrotter(H0, H1, h/2, cs, positive)
    LieTrotter(H1, H0, h/2, cs, positive)


def Yoshida4(H0, H1, h, cs, positive):
    
    uk = 1/(2-2**(1/3))

    Strang(H0, H1, uk*h, cs, positive)
    Strang(H0, H1, (1-2*uk)*h, cs, positive)
    Strang(H0, H1, uk*h, cs, positive)

def Suzuki4(H0, H1, h, cs, positive):
    
    uk = 1/(4-4**(1/3))

    Strang(H0, H1, uk*h, cs, positive)
    Strang(H0, H1, uk*h, cs, positive)
    Strang(H0, H1, (1-4*uk)*h, cs, positive)
    Strang(H0, H1, uk*h, cs, positive)
    Strang(H0, H1, uk*h, cs, positive)

def sym_Zassenhaus4(H0, H1, h):

    cs = get_double_commutator_coefficients(order = 4)

    simulateA = partial(LieTrotter_ff, H=H0)
    simulateB = partial(LieTrotter_ff, H=H1)
    simulateABA = partial(DoubleCommutatorEvolution, simulateA=simulateA, 
                        simulateB=simulateB, h=-h, cs=cs)
    simulateBBA = partial(DoubleCommutatorEvolution, simulateA=simulateB,
                        simulateB=simulateA, h=h, cs=cs)

    simulateB(h = h/2) # B = X
    simulateA(h = h/2) # A = Y
    simulateABA(h = h/np.cbrt(24)) # 2*[Y,[X,Y]]/(24*2)
    simulateBBA(h = h/np.cbrt(24)) # 2*[X,[X,Y]]/48
    simulateABA(h = h/np.cbrt(24)) # 2*[Y,[X,Y]]/(24*2)
    simulateA(h = h/2) # A = Y
    simulateB(h = h/2) # B = X

def get_coefficients(commutator_method):
        if commutator_method == 'NCP_3_6':
            cs, _ = NCP_3_6()
            positive = False
        elif commutator_method == 'NCP_4_10':
            cs, _ = NCP_4_10()
            positive = False
        elif commutator_method == 'PCP_5_16':
            cs, _ = PCP_5_16()
            positive = True
        elif commutator_method == 'PCP_6_26':
            cs, _ = PCP_6_26()
            positive = True
        elif commutator_method == 'PCP_4_12':
            cs, _ = PCP_5_16()
            positive = True
        elif commutator_method == 'NCP_5_18':
            cs, _ = NCP_5_18()
            positive = False
        return cs,positive

def get_double_commutator_coefficients(order):
    if order == 3:
        return [-1, 1, 1, -1, -1, -1, 1, 1]
    elif order == 4:
        d2 = ((np.sqrt(1346)-36)/25)**(1/3)
        d0, d1, d3, d4 = -d2/2, -1/np.sqrt(d2), +1/np.sqrt(d2), -d2
        return [d0, d1, d2, d3, d4, d3, d2, d1, d0]
    else:
        raise ValueError('Order not recognized')


def time_simulation(hamiltonian, time, n_steps, dev, n_wires,
                    n_samples = 20, method = 'LieTrotter', commutator_method = 'NCP_3_6',
                    approximate = True):
    r"""
    Implements Hamiltonian simulation.

    Arguments:
    ---------
    hamiltonian: tuple
        (H0, H1, c)
    time: float
        The total time of the simulation
    n_steps: int
        The number of steps
    n_wires: int
        The number of wires
    n_samples: int
        The number of samples
    method: str
        The method to use for the simulation
    device: str
        The device to use for the simulation
    commutator_method: str
        The method to compute the commutator
    approximate: bool
        Whether to use the approximate method

    Returns:
    --------
    circuit
    """
    @qml.qnode(dev)
    def initial_layer(init_weights, weights):
        # Initial state preparation, using a 2-design
        qml.SimplifiedTwoDesign(initial_layer_weights=init_weights, weights=weights, wires=range(n_wires))
        return qml.state()

    def circuit(time, n_steps, init_state):

        h = time / n_steps

        # Initial state preparation, using a 2-design
        qml.StatePrep(init_state, wires=range(n_wires))

        H0, H1 = hamiltonian[0], hamiltonian[1]

        cs, positive = get_coefficients(commutator_method)

        if method == 'Commutator':
            simulateA = partial(LieTrotter_ff, H=H0)
            simulateB = partial(LieTrotter_ff, H=H1)
            for _ in range(n_steps**2): CommutatorEvolution(simulateA, simulateB, h, cs, positive)
        elif method == 'LieTrotter':
            for _ in range(n_steps): LieTrotter(H0, H1, h, cs, positive)
        elif method == 'Strang':
            for _ in range(n_steps): Strang(H0, H1, h, cs, positive)
        elif method == "Yoshida4":
            for _ in range(n_steps): Yoshida4(H0, H1, h, cs, positive)
        elif method == 'Suzuki4':
            for _ in range(n_steps): Suzuki4(H0, H1, h, cs, positive)
        elif method == 'SymZassenhaus4':
            for _ in range(n_steps): sym_Zassenhaus4(H0, H1, h)
        elif method == 'NestedCommutator':
            cs = get_double_commutator_coefficients(order = 4)
            simulateA = partial(LieTrotter_ff, H=H0)
            simulateB = partial(LieTrotter_ff, H=H1)
            for _ in range(n_steps**3): DoubleCommutatorEvolution(simulateA, simulateB, h, cs)
        else:
            raise ValueError('Method not recognized')

        return qml.state()

    
    def call_approx_full(time, n_steps, init_state):
        resources = qml.resource.get_resources(circuit, gate_set = gs)(time, n_steps, init_state)
        state = qml.QNode(circuit, dev)(time, n_steps, init_state)
        return state, resources

    if approximate:
        average_error = 0.
        for n in tqdm(range(n_samples), desc='Initial states attempted'):
            init_weights = np.random.uniform(0, 2*np.pi, (n_wires,))
            weights = np.random.uniform(0, 2*np.pi, (3, n_wires-1, 2))
            init_state = initial_layer(init_weights, weights)
            st, resources = call_approx_full(time, n_steps, init_state)
            st2, _ = call_approx_full(time, 2*n_steps, init_state)
            average_error = n/(n+1)*average_error + 1/(n+1)*np.linalg.norm(st - st2)
            
    return average_error, resources

def evaluate_commutators(H0, H1):
    r"""Prints some commutators"""
    print('H0:', qml.jordan_wigner(H0), '\n')
    print('H1:', qml.jordan_wigner(H1), '\n')
    CH0H1 = qml.commutator(qml.jordan_wigner(H0), qml.jordan_wigner(H1))
    print('[H0, H1]:', CH0H1, '\n')
    CH0H1H0 = qml.commutator(CH0H1, qml.jordan_wigner(H0))
    print('[H0, [H0, H1]]:', CH0H1H0, '\n')
    CH1H0H1 = qml.commutator(qml.jordan_wigner(H1), CH0H1)
    print('[H1, [H0, H1]]:', CH1H0H1, '\n')
    CH1H0H1H0 = qml.commutator(qml.jordan_wigner(H1), CH0H1H0)
    print('[H1, [H0, [H0, H1]]]:', CH1H0H1H0, '\n')
    CH0H1H0H1 = qml.commutator(qml.jordan_wigner(H0), CH1H0H1)
    print('[H0, [H1, [H0, H1]]]:', CH0H1H0H1, '\n')
    CH0H1H0H1H0 = qml.commutator(qml.jordan_wigner(H0), CH1H0H1H0)
    print('[H0, [H1, [H0, [H0, H1]]]]:', CH0H1H0H1H0, '\n')
    CH1H0H1H0H1 = qml.commutator(qml.jordan_wigner(H1), CH0H1H0H1)
    print('[H1, [H0, [H1, [H0, H1]]]]:', CH1H0H1H0H1, '\n')
    CH1H0H1H0H1H0 = qml.commutator(qml.jordan_wigner(H1), CH0H1H0H1H0)
    print('[H1, [H0, [H1, [H0, [H0, H1]]]]]:', CH1H0H1H0H1H0, '\n')
    CH0H1H0H1H0H1 = qml.commutator(qml.jordan_wigner(H0), CH1H0H1H0H1)
    print('[H0, [H1, [H0, [H1, [H0, H1]]]]]:', CH0H1H0H1H0H1, '\n')

