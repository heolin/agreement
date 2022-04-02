# Agreement [![Build Status](https://travis-ci.com/heolin123/agreement.svg?branch=master)](https://travis-ci.com/heolin123/agreement)

## Inter-rater agreement
Agreement library provides an implementation of popular metrics used to measure inter-rater agreement.
Inter-rater agreement (know also as a inter-rater reliability) is used to describe the degree of agreement among raters.
It is a score of how much homogeneity or consensus exists in the ratings given by various judges.

If you want to learn more about this topic, you can start by reading this [Wikipedia](https://en.wikipedia.org/wiki/Inter-rater_reliability) page.

### Implemented metrics
This library provides a pure numpy implementation of an extended formulas for following metrics:

- **Observed agreement**
- **Bennett et al.'s S score**
- **Cohen's kappa**
- **Gwet's gamma**
- **Krippendorff alpha**
- **Scott's pi**

And extended formulas can be used to measure agreement for;
- **multiple raters** - support for two or more raters,
- **multiple categories** - support for binary problems, as well as more categories,
- **missing ratings** - not all raters provided answers for all the questions.
- **weighted agreement** - used to model distance between categories (e.g. `dist(5, 4) < dist(5, 1)`)

More information about implemented metrics can be found here: TODO

### Implemented weights kernels
Agreement provides implementations for eight weight kernels:

- **identity kernel** 
- **linear kernel**
- **quadratic kernel**
- **ordinal_kernel**
- **radical_kernel**
- **radio_kernel**
- **circular_kernel**
- **bipolar_kernel**

More information about implemented weights kernels can be found here: TODO

## Installation
Agreement can be installed via pip from [PyPI](https://pypi.org/project/agreement/).
 
```bash
pip install agreement
```

## Example usage
#### 1. Prepare dataset
Let's assume you have a dataset in a format of a matrix with three columns: `question id`, `rater id` and `answer`. 

```python
import numpy  as np

dataset = np.array([
    [1, 1, 'a'],
    [1, 2, 'a'],
    [1, 3, 'c'],
    [2, 1, 'a'],
    [2, 2, 'b'],
    [2, 3, 'a'],
    [3, 1, 'c'],
    [3, 2, 'b'],
])
```
#### 2. Transform dataset into matrices
In the next step we want to transform the dataset into matrices in a form accepted by the metrics functions.

Most of the matrices require a "questions answers" matrix, which contains a frequency of answers for each question.
So more formally we could say `M = I x A`, where `I` is a list of all items and `A` is a list of all possible answers.
Matrix element `M_ij` represents how many times answer `j` was chosen for the questions `i`.

The second matrix can be required (currently it is only required by the Cohen's kappa metrics) is "users answers" matrix, which
contains a frequency of answers selected by each user.
So more formally we could say `M = U x A`, where `U` is a list of all users and `A` is a list of all possible answers.
Matrix element `M_ij` represents how many times answer `j` was chosen for the user `i`.

The library provides a helper functions that can be used to prepare that.

```python
from agreement.utils.transform import pivot_table_frequency

questions_answers_table = pivot_table_frequency(dataset[:, 0], dataset[:, 2])
users_answers_table = pivot_table_frequency(dataset[:, 1], dataset[:, 2])
```

#### 3. Select kernel
Weights are used to model situations, where categories are represented as (at least) ordinal data.
Using this approach, the agreement between raters is not binary, but it differs depending on the
weights between chosen categories.

There is no formal rule that can be used for deciding which set weights should be used
in a particular study, so it all depends on your problem and the data your are working with.

In a default, metrics are using the `identity_kernel`, which do not provide any weighting between the answers.
If you want to use an alternative kernel, you can import it from:

```python
from agreement.utils.kernels import linear_kernel
```

#### 4. Compute the metric
The last step is to chose the metric you want to compute and run following code:
```python
from agreement.metrics import cohens_kappa, krippendorffs_alpha

kappa = cohens_kappa(questions_answers_table, users_answers_table)
weighted_kappa = cohens_kappa(questions_answers_table, users_answers_table, weights_kernel=linear_kernel)

alpha = krippendorffs_alpha(questions_answers_table)
```

For more detailed example see: TODO

## Reference
All equations are based on the Handbook of Inter-Rater ReLiability, Kilem Li. Gwet, 2014.
This book provides an extensive explanation to all topics related to inter-rater agreement.
The book provides a detailed description of all metrics implemented in this library, as well as 
an example datasets that were used to this this implementation.

I also recommend taking a look at MatLab implementation of the same metrics [mReliability](https://github.com/jmgirard/mReliability),
which provides a more detailed explanation of metrics' formulas then the one you will find here.

