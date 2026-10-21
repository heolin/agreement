from agreement.metrics import (
    observed_agreement,
    s_score,
    cohens_kappa,
    gwets_gamma,
    krippendorffs_alpha,
    scotts_pi
)
from agreement.utils.kernels import quadratic_kernel


def test_observed_agreement(dataset_gwet_transformed):
    questions_answers_table, _ = dataset_gwet_transformed
    po = observed_agreement(questions_answers_table)
    assert round(po, 4) == 0.8182


def test_s_score(dataset_gwet_transformed):
    questions_answers_table, _ = dataset_gwet_transformed
    s = s_score(questions_answers_table)
    assert round(s, 4) == 0.7727


def test_cohens_kappa(dataset_gwet_transformed):
    questions_answers_table, users_answers_table = dataset_gwet_transformed
    k = cohens_kappa(questions_answers_table, users_answers_table)
    assert round(k, 4) == 0.7628


def test_gwets_gamma(dataset_gwet_transformed):
    questions_answers_table, _ = dataset_gwet_transformed
    g = gwets_gamma(questions_answers_table)
    assert round(g, 4) == 0.7754


def test_krippendorffs_alpha(dataset_gwet_transformed):
    questions_answers_table, _ = dataset_gwet_transformed
    a = krippendorffs_alpha(questions_answers_table)
    assert round(a, 4) == 0.7434


def test_krippendorffs_alpha_quadratic(dataset_gwet_transformed):
    # Expected value from the `krippendorff` package (0.8.2), level_of_measurement="interval".
    questions_answers_table, _ = dataset_gwet_transformed
    a = krippendorffs_alpha(questions_answers_table, weights_kernel=quadratic_kernel)
    assert round(a, 4) == 0.8491


def test_scotts_pi(dataset_gwet_transformed):
    questions_answers_table, _ = dataset_gwet_transformed
    pi = scotts_pi(questions_answers_table)
    assert round(pi, 4) == 0.7625
