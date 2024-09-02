import pennylane as qml
import numpy as np
from tqdm import tqdm
from scipy.linalg import logm

from coefficients.generate_coefficients import NCP_3_6, NCP_4_10, NCP_5_18, PCP_5_16, PCP_6_26

# You can choose from BoseHubbard, FermiHubbard, Heisenberg, Ising, check https://pennylane.ai/datasets/
def load_hamiltonian(name = "Ising", periodicity="open", lattice="chain", layout="1x4"):

    hamiltonians = qml.data.load("qspin", sysname=name, periodicity=periodicity, 
                                lattice=lattice, layout=layout, attributes=["hamiltonians"])

    return hamiltonians[0].hamiltonians[50]

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
    qml.exp(H, h, 1) #todo: change back to 
    #qml.TrotterProduct(H, h)


def CommutatorEvolution(H0, H1, h, coupling, cs, positive = True):
    r"""
    Implements the evolution of the commutator term in the LieTrotter step.

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
    cs: np.array
        The coefficients of the commutator
    positive: bool
        Whether to use positive or negative counte-palyndromic coefficients
    """
    m = len(cs)-1

    cs_ = cs * (-1)**(int(positive) + 1)
    for c0, c1 in zip(cs_[::2], cs_[1::2]):
        LieTrotter_ff(H0, h*coupling*c0)
        LieTrotter_ff(H1, h*coupling*c1)
    if m%2 == 0: 
        LieTrotter_ff(H0, h*coupling*cs_[-1])

    if m%2 == 0: H0, H1 = H1, H0

    cs_ = cs[::-1]
    for c0, c1 in zip(cs_[::2], cs_[1::2]):
        LieTrotter_ff(H0, h*coupling*c0)
        LieTrotter_ff(H1, h*coupling*c1)
    if m%2 == 0: 
        LieTrotter_ff(H0, h*coupling*cs_[-1])


def LieTrotter(H0, H1, h, coupling, cs, positive, reversed = False):
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

    if not reversed:
        LieTrotter_ff(H0, h)
        LieTrotter_ff(H1, h)
        CommutatorEvolution(H0, H1, h, coupling, cs, positive)
    else:
        CommutatorEvolution(H0, H1, h, coupling, cs, positive)
        LieTrotter_ff(H1, h)
        LieTrotter_ff(H0, h)

def Strang(H0, H1, h, coupling, cs, positive):
    LieTrotter(H0, H1, h/2, coupling, cs, positive)
    LieTrotter(H0, H1, h/2, coupling, cs, positive, reversed = True)

def Suzuki4(H0, H1, h, coupling, cs, positive):
    
    uk = 1/(4-4**(1/3))

    Strang(H0, H1, uk*h, coupling, cs, positive)
    Strang(H0, H1, uk*h, coupling, cs, positive)
    Strang(H0, H1, (1-4*uk)*h, coupling, cs, positive)
    Strang(H0, H1, uk*h, coupling, cs, positive)
    Strang(H0, H1, uk*h, coupling, cs, positive)


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


def basic_simulation(hamiltonian, time, n_steps, dev, n_wires,
                    n_samples = 3, method = 'LieTrotter', commutator_method = 'NCP_3_6',
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
    def call_approx(time, n_steps, init_weights, weights):

        h = time / n_steps

        # Initial state preparation, using a 2-design
        qml.SimplifiedTwoDesign(initial_layer_weights=init_weights, weights=weights, wires=range(n_wires))

        H0, H1, coupling = hamiltonian[0], hamiltonian[1], hamiltonian[2]

        cs, positive = get_coefficients(commutator_method)

        if method == 'Commutator':
            for _ in range(n_steps): CommutatorEvolution(H0, H1, h, coupling, cs, positive)
        elif method == 'LieTrotter':
            for _ in range(n_steps): LieTrotter(H0, H1, h, coupling, cs, positive)
        elif method == 'Strang':
            for _ in range(n_steps): Strang(H0, H1, h, coupling, cs, positive)
        elif method == 'Suzuki4':
            for _ in range(n_steps): Suzuki4(H0, H1, h, coupling, cs, positive)
        else:
            raise ValueError('Method not recognized')

        return qml.state()

    
    def call_approx_full(time, n_steps, init_weights, weights):
        state = call_approx(time, n_steps, init_weights, weights)
        return state

    if approximate:
        average_error = 0.
        for n in tqdm(range(n_samples), desc='Initial states attempted'):
            init_weights = np.random.uniform(0, 2*np.pi, (n_wires,))
            weights = np.random.uniform(0, 2*np.pi, (3, n_wires-1, 2))
            average_error = n/(n+1)*average_error + 1/(n+1)*np.linalg.norm(call_approx_full(time, n_steps, init_weights, weights)
                                                        - call_approx_full(time, 5*n_steps, init_weights, weights))
    else:
        cs, positive = get_coefficients(commutator_method)
        H0, H1, coupling = hamiltonian[0], hamiltonian[1], hamiltonian[2]
        h = time / n_steps

        app = logm(qml.matrix(CommutatorEvolution, wire_order=range(n_wires))(H0, H1, h, coupling, cs, positive))
        ex = logm(qml.matrix(qml.exp(qml.commutator(H0, H1), (h*coupling)**2), wire_order=range(n_wires)))
        average_error = np.linalg.norm(app - ex)

    return average_error

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

