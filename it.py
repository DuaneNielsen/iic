import torch


def entropy(p, dim=None):
    eps = torch.finfo(p.dtype).eps
    if dim is None:
        return - torch.sum(p * torch.log2(p + eps))
    else:
        return - torch.sum(p * torch.log2(p + eps), dim=dim)


def conditional(P, given, transpose=False):
    """

    :param P: a joint distribution P(X, Y)
    :param given: the dimension to condition on
    :param transpose: transposes the result such that first dim contains probability distributions
    ie you can always get P(Y, x=0) by returning conditional(joint, 0, transpose=True)[0]
    likewise you can get P(X, y=0) by conditional(joint, 1, transpose=True)[0]
    :return: all

    see https://www.youtube.com/watch?v=9w4LnXIip5A&list=TLPQMDcwMjIwMjDBb2j84xxgjw&index=2 42:30 for a worked example
    """
    """ Returns P(X|Y) where Y is specified by dim """

    eps = torch.finfo(P.dtype).eps
    C = P / (torch.sum(P, dim=given, keepdim=True) + eps)
    if transpose:
        names = list(P.names)
        names.remove(given)
        names.append(given)
        names = tuple(names)
        C = C.align_to(*names)

    return C


def conditional_entropy(P, given, index=None):
    """

    :param P: the joint distrubution
    :param given: the variable that is known
    :param index: optional: if set to the index of the known variable, will return the entropy conditional on the known
    variable being equal to the value
    :return:
    conditional_entropy(P, 'Y') returns H(X|Y)
    conditional_entropy(P, 'Y', 0) returns H(X|Y == 0)
    """
    eps = torch.finfo(P.dtype).eps
    if index is not None:
        C = conditional(P, given, transpose=True)
        H = - C * torch.log2(C + eps)
        H = torch.sum(H, dim=1)
        return H[index]
    else:
        C = conditional(P, given)
        return - torch.sum(P * torch.log2(C + eps))


def marginal(P, dim, keepdim=False):
    """

    :param P: a named tensor with the joint distribution
    :param dim: the dimension to compute the marginal for
    :param keepdim: if true, retains the dimensions
    :return: the marginal distribution of dim
    """
    return torch.sum(P, dim, keepdim)


def mutual_information(P):
    eps = torch.finfo(P.dtype).eps
    marginal_product = marginal(P, P.names[0], keepdim=True) * marginal(P, P.names[1], keepdim=True) + eps
    H = P * torch.log2((P / marginal_product) + eps)
    return torch.sum(H)


def mutual_information_stable(P):
    """is this actually more numerically stable than the other... I don't know..."""
    """definately it's less accurate"""
    eps = torch.finfo(P.dtype).eps
    P = P.rename(None)
    P[P < eps] = eps
    logP= torch.log2(P)
    logM1, logM2 = torch.log2(torch.sum(P, dim=0, keepdim=True)), torch.log2(torch.sum(P, dim=1, keepdim=True))
    return torch.sum(P * (logP - logM1 - logM2))

