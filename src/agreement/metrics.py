import numpy as np
from agreement.utils.kernels import (
    identity_kernel, get_weights
)


def observed_agreement(df, weights_kernel=identity_kernel):
    N, q = df.shape
    n = df.sum(axis=1)

    w = get_weights(q, weights_kernel)

    r_star = df.dot(w)

    po = ((df * (r_star-1)).sum(axis=1) / (n * (n-1))).sum()/N
    return po


def s_score(df, weights_kernel=identity_kernel):
    N, q = df.shape
    n = df.sum(axis=1)

    w = get_weights(q, weights_kernel)

    r_star = df.dot(w)

    po = ((df * (r_star-1)).sum(axis=1) / (n * (n-1))).sum()/N

    pc = w.sum() / q**2

    S = (po - pc) / (1 - pc)
    return S


def cohens_kappa(df, dfa, weights_kernel=identity_kernel):
    N, q = df.shape
    n = df.sum(axis=1)

    w = get_weights(q, weights_kernel)

    r_star = df.dot(w)

    po = ((df * (r_star-1)).sum(axis=1) / (n * (n-1))).sum() / N

    p = dfa.div(dfa.sum(axis=1), axis=0)
    r, q = p.shape

    pbar = p.sum(axis=0) / r

    rpbar = r * np.array(pbar).reshape(1, q).T * np.array(pbar).reshape(1, q)
    pg = np.array(p).T.dot(np.array(p))
    s2 = (pg - rpbar) / (r - 1)

    pbarplus = np.array(pbar).reshape(1, q).T * np.array(pbar).reshape(1, q)
    pc = (w * (pbarplus - s2/r)).sum()

    k = (po - pc) / (1 - pc)
    return k


def gwets_gamma(df, weights_kernel=identity_kernel):
    N, q = df.shape
    n = df.sum(axis=1)

    w = get_weights(q, weights_kernel)

    r_star = df.dot(w)

    po = ((df * (r_star-1)).sum(axis=1) / (n * (n-1))).sum() / N

    Tw = w.sum()

    pi = df.div(df.sum(axis=1), axis=0).sum() / N

    pc = (pi*(1 - pi)).values.sum() * Tw / (q*(q-1))

    gamma = (po - pc) / (1 - pc)
    return gamma


def krippendorffs_alpha(df, weights_kernel=identity_kernel):
    N, q = df.shape
    n = df.sum(axis=1)

    w = get_weights(q, weights_kernel)

    r_star = df.dot(w)

    po2 = ((df * (r_star-1)).sum(axis=1) / (n * (n-1))).sum()/N

    rdash = df.sum(axis=1).sum() / N

    epsilon = 1 / (N*rdash)
    po = po2 * (1-epsilon) + epsilon

    pi = df.div(df.sum(axis=1), axis=0).sum() / N

    pc = (w * np.array(pi).reshape(1, q).T * np.array(pi).reshape(1, q)).sum()

    alpha = (po - pc) / (1 - pc)
    return alpha


def scotts_pi(df, weights_kernel=identity_kernel):
    N, q = df.shape
    n = df.sum(axis=1)

    w = get_weights(q, weights_kernel)

    r_star = df.dot(w)

    po = ((df * (r_star-1)).sum(axis=1) / (n * (n-1))).sum() / N

    pik = df.div(df.sum(axis=1), axis=0).sum() / N

    pc = (w * np.array(pik).reshape(1, q).T * np.array(pik).reshape(1, q)).sum()

    pi = (po - pc) / (1 - pc)
    return pi
