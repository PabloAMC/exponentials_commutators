import pennylane as qml
from pennylane import numpy as np
from tqdm import tqdm

from coefficients.generate_coefficients import NCP_3_6, NCP_4_10, NCP_5_18, PCP_5_16, PCP_6_26

def fermion_chain_1d(n):

    H0 = qml.FermiC(0) * qml.FermiA(1) + qml.FermiC(1) * qml.FermiA(0)
    H1 = qml.FermiC(1) * qml.FermiA(2) + qml.FermiC(2) * qml.FermiA(1)

    for i in range(2, n-1, 2):
        H0 += qml.FermiC(i) * qml.FermiA(i+1) + qml.FermiC(i+1) * qml.FermiA(i)
        H1 += qml.FermiC((i+1)%n) * qml.FermiA((i+2)%n) + qml.FermiC((i+2)%n) * qml.FermiA((i+1)%n)

    return H0, H1

H0, H1 = fermion_chain_1d(4)


def LieTrotter_ff(H, h): 
    qml.TrotterProduct(H, h)
    #for coeff, op in zip(H.coeffs, H.ops):
    #    qml.exp(-1j * h * coeff * op)

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



def basic_simulation(hamiltonian, time, n_steps, dev, n_wires,
                    n_samples = 3, method = 'LieTrotter', commutator_method = 'NCP_3_6'):
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

    error = 0
    for n in tqdm(range(n_samples), desc='Initial states attempted'):
        init_weights = np.random.uniform(0, 2*np.pi, (n_wires,))
        weights = np.random.uniform(0, 2*np.pi, (3, n_wires-1, 2))
        average_error = n/(n+1)*error + 1/(n+1)*np.linalg.norm(call_approx_full(time, n_steps, init_weights, weights)
                                                    - call_approx_full(time, 5*n_steps, init_weights, weights))

    return average_error
