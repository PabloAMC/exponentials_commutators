from functools import partial
import pennylane as qml
import numpy as np
from tqdm import tqdm
from scipy.linalg import logm

import copy
gs = copy.copy(qml.resource.resource.StandardGateSet)
gs.add('StatePrep')

from coefficients.generate_coefficients import NCP_3_6, NCP_4_10, NCP_5_18, PCP_5_16, PCP_6_26, S3_coeffs

# You can choose from BoseHubbard, FermiHubbard, Heisenberg, Ising, check https://pennylane.ai/datasets/
def load_hamiltonian(name = "Ising", periodicity="open", lattice="chain", layout="1x4"):

    hamiltonians = qml.data.load("qspin", sysname=name, periodicity=periodicity, 
                                lattice=lattice, layout=layout, attributes=["hamiltonians"])

    return hamiltonians[0].hamiltonians[50], eval(layout.replace('x', '*'))

def split_hamiltonian(H):
    H0 = 0*qml.Identity(0)
    H1 = 0*qml.Identity(0)

    for coeff, op in zip(H.coeffs, H.ops):
        # We will fix coeff to 1 for all cases.
        if 'X' in str(op.pauli_rep):
            H0 = H0 + op
        else:
            H1 = H1 + op
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


def Commutator(H0, H1, h, cs, positive = True):
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
        LieTrotter_ff(H0, h*c0)
        LieTrotter_ff(H1, h*c1)
    if m%2 == 0: 
        LieTrotter_ff(H0, h*cs_[-1])

    if m%2 == 0: H0, H1 = H1, H0

    cs_ = cs[::-1]
    for c0, c1 in zip(cs_[::2], cs_[1::2]):
        LieTrotter_ff(H0, h*c0)
        LieTrotter_ff(H1, h*c1)
    if m%2 == 0: 
        LieTrotter_ff(H0, h*cs_[-1])


def RecursiveCommutator(S, h, method, order):

    #RC = lambda h: RecursiveCommutator(S, h, method, order-2)

    if order%2 == 0: raise ValueError('The order must be odd')
    if order == 3: S(h, adjoint = False)
    elif order == 5:
        if method == 'G4': G4(S, h, order-2)
        elif method == 'G5': G5(S, h, order-2)
        elif method == 'G6': G6(S, h, order-2)
        elif method == 'G10': G10(S, h, order-2)
        else: raise ValueError('Method not recognized')
    else: raise ValueError('Order not implemented')


def S3(h, adjoint, H0, H1, cs):
    r"""
    Implements the S3 formula

    Arguments:
    ---------
    S: The O(n) step
    h: The time step
    cs: The coefficients
    """


    if not adjoint:
        cs_ = np.array(cs[::-1])
        for cB, cA in zip(cs_[::2], cs_[1::2]):
            LieTrotter_ff(H1, h*cB)
            LieTrotter_ff(H0, h*cA)
    else:
        cs_ = -1*np.array(cs)
        for cA, cB in zip(cs_[::2], cs_[1::2]):
            LieTrotter_ff(H0, h*cA)
            LieTrotter_ff(H1, h*cB)


def G6(S, h, n):
    r"""
    Implements the sqrt(6)-copy formula
    
    Arguments:
    ---------
    S: The O(n) step
    h: The time step
    n: The order
    """
    u = 1/np.sqrt(2-2**(2/(n+1)))
    v = 2**(1/(n+1)) * u

    S(h = -u*h/np.sqrt(2), adjoint = False)
    S(h = -v*h/np.sqrt(2), adjoint = True)
    S(h = -u*h/np.sqrt(2), adjoint = False)
    S(h = u*h/np.sqrt(2), adjoint = False)
    S(h = v*h/np.sqrt(2), adjoint = True)
    S(h = u*h/np.sqrt(2), adjoint = False)


def G10(S, h, n):
    r"""
    Implements the sqrt(10)-copy formula

    Arguments:
    ---------
    S: The O(n) step
    h: The time step
    n: The order
    """
    sigma = 4**(2/(n+1))/(4*(4-4**(2/(n+1))))
    mu = (4*sigma)**(1/2)
    nu = (1/4 + sigma)**(1/2)

    if n%2 == 1:
        S(h = -nu*h/np.sqrt(2), adjoint = False)
        S(h = -nu*h/np.sqrt(2), adjoint = False)
        S(h = -mu*h/np.sqrt(2), adjoint = True)
        S(h = -nu*h/np.sqrt(2), adjoint = False)
        S(h = -nu*h/np.sqrt(2), adjoint = False)

        S(h = nu*h/np.sqrt(2), adjoint = False)
        S(h = nu*h/np.sqrt(2), adjoint = False)
        S(h = mu*h/np.sqrt(2), adjoint = True)
        S(h = nu*h/np.sqrt(2), adjoint = False)
        S(h = nu*h/np.sqrt(2), adjoint = False)

    else:
        S(h = -nu*h/np.sqrt(2), adjoint = False)
        S(h = +nu*h/np.sqrt(2), adjoint = False)

        S(h = -nu*h/np.sqrt(2), adjoint = False)
        S(h = +nu*h/np.sqrt(2), adjoint = False)

        S(h = -mu*h/np.sqrt(2), adjoint = True)
        S(h = +mu*h/np.sqrt(2), adjoint = True)

        S(h = -nu*h/np.sqrt(2), adjoint = False)
        S(h = +nu*h/np.sqrt(2), adjoint = False)

        S(h = -nu*h/np.sqrt(2), adjoint = False)
        S(h = +nu*h/np.sqrt(2), adjoint = False)


def G5(S, h, n):
    r"""
    Implements the sqrt(5)-copy formula

    Arguments:
    ---------
    S: The O(n) step
    h: The time step
    n: The order
    """
    s = (2/(1+2**(1/(n+2))))**(1/(n+1))
    sp = 2**(-1/(n+2))*s

    S(h = -sp*h, adjoint = False)
    S(h = -h, adjoint = True)
    S(h = s*h, adjoint = False)
    S(h = +h, adjoint = True)
    S(h = -sp*h, adjoint = False)

def G4(S, h, n):
    r"""
    Implements the sqrt(4)-copy formula

    Arguments:
    ---------
    S: The O(n) step
    h: The time step
    n: The order
    """

    a = 1
    b = 2
    if n == 3:
        c = 1.982590733
        d = -0.8190978288
    elif n == 5:
        c = 1.996950166
        d = -0.8642318466
    elif n == 7:
        c = 1.999411381
        d = -0.8911860667
    elif n == 9:
        c = 1.999880034
        d = -0.9091844711
    elif n == 11:
        c = 1.999974677
        d = -0.9220693131

    s = np.sqrt(np.abs(a**2 - b**2 + c**2 - d**2))

    S(h = d/s*h, adjoint = True)
    S(h = c/s*h, adjoint = False)
    S(h = b/s*h, adjoint = True)
    S(h = a/s*h, adjoint = False)


def LieTrotter(H0, H1, h):
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

    Returns:
    --------
    None
    """
    LieTrotter_ff(H0, h)
    LieTrotter_ff(H1, h)


def Strang(H0, H1, h):
    LieTrotter(H0, H1, h/2)
    LieTrotter(H1, H0, h/2)

def Suzuki4(H0, H1, h):
    
    uk = 1/(4-4**(1/3))

    Strang(H0, H1, uk*h)
    Strang(H0, H1, uk*h)
    Strang(H0, H1, (1-4*uk)*h)
    Strang(H0, H1, uk*h)
    Strang(H0, H1, uk*h)


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



def time_simulation(hamiltonian, time, n_steps, dev, n_wires,
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
    def initial_layer(init_weights, weights):
        # Initial state preparation, using a 2-design
        qml.SimplifiedTwoDesign(initial_layer_weights=init_weights, weights=weights, wires=range(n_wires))
        return qml.state()

    def circuit(time, n_steps, init_state):

        h = time / n_steps

        # Initial state preparation, using a 2-design
        qml.StatePrep(init_state, wires=range(n_wires))

        H0, H1, _ = hamiltonian[0], hamiltonian[1], hamiltonian[2]

        if method == 'Commutator':
            cs, positive = get_coefficients(commutator_method)
            for _ in range(n_steps**2): Commutator(H0, H1, h, cs, positive)
        elif method == 'RecursiveCommutator':
            cs, _ = S3_coeffs()
            S = partial(S3, H0 = H0, H1 = H1, cs = cs)
            cm, order = commutator_method.split('_')[0], int(commutator_method.split('_')[1])
            for _ in range(n_steps**2): RecursiveCommutator(S, h, cm, order)
        elif method == 'LieTrotter':
            for _ in range(n_steps): LieTrotter(H0, H1, h)
        elif method == 'Strang':
            for _ in range(n_steps): Strang(H0, H1, h)
        elif method == 'Suzuki4':
            for _ in range(n_steps): Suzuki4(H0, H1, h)
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
    else:
        cs, positive = get_coefficients(commutator_method)
        H0, H1, coupling = hamiltonian[0], hamiltonian[1], hamiltonian[2]
        h = time / n_steps

        app = logm(qml.matrix(Commutator, wire_order=range(n_wires))(H0, H1, h, coupling, cs, positive))
        ex = logm(qml.matrix(qml.exp(qml.commutator(H0, H1), (h*coupling)**2), wire_order=range(n_wires)))
        average_error = np.linalg.norm(app - ex)
        resources = 1

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

