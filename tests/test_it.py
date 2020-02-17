from iic.it import *
import torch
from torch import allclose
import train_classifier

P = torch.tensor([
    [1 / 8, 1 / 16, 1 / 32, 1 / 32],
    [1 / 16, 1 / 8, 1 / 32, 1 / 32],
    [1 / 16, 1 / 16, 1 / 16, 1 / 16],
    [1 / 4, 0, 0, 0]
], names=('X', 'Y'))


def test_conditional():
    P_X_Y = torch.tensor([
        [1 / 2, 1 / 4, 1 / 8, 1 / 8],
        [1 / 4, 1 / 2, 1 / 8, 1 / 8],
        [1 / 4, 1 / 4, 1 / 4, 1 / 4],
        [1.0, 0.0, 0.0, 0.0]
    ], names=('X', 'Y'))

    P_eps = torch.finfo(P.dtype).eps
    assert allclose(conditional(P,'Y').rename(None), P_X_Y.rename(None), rtol=0.0, atol=P_eps * 4)
    assert allclose(conditional(P,'Y', transpose=True).rename(None), P_X_Y.rename(None), rtol=0.0, atol=P_eps * 4)

    P_Y_X = torch.tensor([
        [1 / 4, 1 / 8, 1 / 8, 1 / 2],
        [1 / 4, 1 / 2, 1 / 4, 0.0],
        [1 / 4, 1 / 4, 1 / 2, 0.0],
        [1 / 4, 1 / 4, 1 / 2, 0.0]
    ], names=('X', 'Y'))
    assert allclose(conditional(P,'X').rename(None), P_Y_X.rename(None).T, rtol=0.0, atol=P_eps * 4)
    assert allclose(conditional(P,'X', transpose=True).rename(None), P_Y_X.rename(None), rtol=0.0, atol=P_eps * 4)


def close(x, y, distance=35):
    eps = torch.finfo(type=torch.float32).eps
    return abs(x - y) < eps * distance


def test_marginal_entropy():
    assert close(entropy(P).item(), 27 / 8)
    assert close(entropy(marginal(P, 'X')).item(), 7 / 4)
    assert close(entropy(marginal(P, 'Y')).item(), 2.0)


def test_conditional_entropy():
    assert close(conditional_entropy(P, 'Y', 0).item(), 7 / 4)
    assert close(conditional_entropy(P, 'Y', 1).item(), 7 / 4)
    assert close(conditional_entropy(P, 'Y', 2).item(), 2.0)
    assert close(conditional_entropy(P, 'Y', 3).item(), 0)

    assert close(conditional_entropy(P, 'X', 0).item(), 7 / 4)
    assert close(conditional_entropy(P, 'X', 1).item(), 6 / 4)
    assert close(conditional_entropy(P, 'X', 2).item(), 6 / 4)
    assert close(conditional_entropy(P, 'X', 3).item(), 6 / 4)

    assert close(conditional_entropy(P, 'Y').item(), 11 / 8)
    assert close(conditional_entropy(P, 'X').item(), 13 / 8)


def test_mutual_information():
    H = entropy(P)
    H_X = entropy(marginal(P, 'X'))
    H_Y = entropy(marginal(P, 'Y'))
    H_X_given_Y = conditional_entropy(P, 'Y')
    H_Y_given_X = conditional_entropy(P, 'X')

    I = {}

    I['I_XY'] = mutual_information(P)
    I['I_XY_st'] = mutual_information_stable(P)
    I['mi'] = train_classifier.mi(P.rename(None))
    I['I_XY_x'] = H_X - H_X_given_Y
    I['I_YX_y'] = H_Y - H_Y_given_X
    I['I_XY_m'] = H_X + H_Y - H
    I['I_XY_c'] = H - H_X_given_Y - H_Y_given_X

    eps = torch.finfo(torch.float32).eps
    mp = marginal(P, 'X', keepdim=True) * marginal(P, 'Y', keepdim=True) + eps
    c = P / mp
    I['I_XY_d'] = torch.sum((c + eps).log2() * P)

    # marginal_product = marginal(P, P.names[0], keepdim=True) * marginal(P, P.names[1], keepdim=True) + eps
    # H = P * torch.log2((P / marginal_product) + eps)
    # return torch.sum(H)

    #print(I)

    for i, mi_i in I.items():
        for j, mi_j in I.items():
            msg = f'{i} : {mi_i}, {j}: {mi_j}, dist: {abs(mi_i - mi_j)/eps}'
            print(msg)
            if not close(mi_i, mi_j, distance=100):
                print(msg)
                assert False
