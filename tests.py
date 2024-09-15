import numpy as np
import pennylane as qml

from coefficients.generate_coefficients import NCP_3_6, NCP_4_10
from pennylane_subroutines_commutator import LieTrotter, LieTrotter_ff, CommutatorEvolution, fermion_chain_1d

def commutator_technique(commutator_method):
    if commutator_method == 'NCP_3_6':
        cs, _ = NCP_3_6()
        positive = False
    elif commutator_method == 'NCP_4_10':
        cs, _ = NCP_4_10()
        positive = False
    return cs, positive

def test_commutator(H0, H1, coupling, commutator_method, wire_order):

    cs, positive = commutator_technique(commutator_method)

    CM = qml.matrix(CommutatorEvolution, wire_order=wire_order)(H0, H1, h, coupling, cs, positive)

    M1 = qml.exp(-1j * h * H0)
    M2 = qml.exp(-1j * h * H1)

def testLieTrotter(H0, H1, wire_order):

    M0 = qml.matrix(LieTrotter_ff, wire_order=wire_order)(H0, h)
    M1 = qml.matrix(LieTrotter_ff, wire_order=wire_order)(H1, h)

    M = qml.matrix(LieTrotter, wire_order=wire_order)(H0, H1, h, None, 'NCP_3_6')

    assert np.allclose(M, M1 @ M0)

modes = 4
H0, H1 = fermion_chain_1d(modes)
coupling = 0.5
h = 0.1

testLieTrotter(qml.jordan_wigner(H0), qml.jordan_wigner(H1), range(modes))

#test_commutator(qml.jordan_wigner(H0), qml.jordan_wigner(H1), coupling, 'NCP_3_6', wire_order=range(modes))