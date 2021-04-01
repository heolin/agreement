import numpy as np
from agreement.utils.kernels import (
    identity_kernel, compute_weights
)


def _compute_observed_agreement(answers_matrix, weights_kernel=identity_kernel):
    N, q = answers_matrix.shape
    n = answers_matrix.sum(axis=1)

    w = compute_weights(q, weights_kernel)

    r_star = answers_matrix.dot(w)

    po = np.average((answers_matrix * (r_star-1)).sum(axis=1) / (n * (n-1)))
    return po, w


def observed_agreement(answers_matrix, weights_kernel=identity_kernel):
    answers_matrix = answers_matrix[answers_matrix.sum(axis=1) > 1]
    po, _ = _compute_observed_agreement(answers_matrix, weights_kernel)
    return po


def s_score(answers_matrix, weights_kernel=identity_kernel):
    answers_matrix = answers_matrix[answers_matrix.sum(axis=1) > 1]
    N, q = answers_matrix.shape

    po, w = _compute_observed_agreement(answers_matrix, weights_kernel)

    pc = w.sum() / q**2

    S = (po - pc) / (1 - pc)
    return S


def cohens_kappa(answers_matrix, users_matrix, weights_kernel=identity_kernel):
    answers_matrix = answers_matrix[answers_matrix.sum(axis=1) > 1]

    N, q = answers_matrix.shape

    po, w = _compute_observed_agreement(answers_matrix, weights_kernel)

    p = users_matrix / users_matrix.sum(axis=1, keepdims=True)

    r, q = p.shape

    pbar = p.sum(axis=0) / r

    rpbar = r * pbar.reshape(1, q).T * pbar.reshape(1, q)
    pg = p.T.dot(p)
    s2 = (pg - rpbar) / (r - 1)

    pbarplus = pbar.reshape(1, q).T * pbar.reshape(1, q)
    pc = (w * (pbarplus - s2/r)).sum()

    k = (po - pc) / (1 - pc)
    return k


def gwets_gamma(answers_matrix, weights_kernel=identity_kernel):
    N, q = answers_matrix.shape

    _answers_matrix = answers_matrix[answers_matrix.sum(axis=1) > 1]
    po, w = _compute_observed_agreement(_answers_matrix, weights_kernel)

    Tw = w.sum()

    pi = np.average(answers_matrix / answers_matrix.sum(axis=1, keepdims=True), axis=0)

    pc = (pi*(1 - pi)).sum() * Tw / (q*(q-1))

    gamma = (po - pc) / (1 - pc)
    return gamma



def krippendorffs_alpha(answers_matrix, weights_kernel=identity_kernel):
    _answers_matrix = answers_matrix[answers_matrix.sum(axis=1) > 1]

    N, q = _answers_matrix.shape
    n = _answers_matrix.sum(axis=1)

    w = compute_weights(q, weights_kernel)

    r_star = _answers_matrix.dot(w)

    rdash = _answers_matrix.sum(axis=1).sum() / N
    epsilon = 1 / (N * rdash)

    po2 = (_answers_matrix * (r_star-1) / (rdash * (n - 1))[:,np.newaxis]).sum(axis=1).mean()

    po = po2 * (1 - epsilon) + epsilon

    pc = ((_answers_matrix.mean(axis=0) / rdash)**2).sum()

    alpha = (po - pc) / (1 - pc)
    return alpha


def scotts_pi(answers_matrix, weights_kernel=identity_kernel):
    answers_matrix = answers_matrix[answers_matrix.sum(axis=1) > 1]
    N, q = answers_matrix.shape

    po, w = _compute_observed_agreement(answers_matrix, weights_kernel)

    pik = np.mean(answers_matrix / answers_matrix.sum(axis=1, keepdims=True), axis=0)

    pc = (w * pik.reshape(1, q).T * pik.reshape(1, q)).sum()

    pi = (po - pc) / (1 - pc)
    return pi

