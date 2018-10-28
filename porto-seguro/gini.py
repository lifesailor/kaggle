import numpy as np


def gini(actual, pred):
    assert (len(actual) == len(pred))

    # all column: actual, pred, range
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)

    # sort all: pred, range.
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]

    # actual
    totalLosses = all[:, 0].sum()

    # actual cumsum
    giniSum = all[:, 0].cumsum().sum() / totalLosses
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_normalized(actual, pred):
    return gini(actual, pred) / gini(actual, actual)


if __name__ == "__main__":
    predictions = [0.9, 0.3, 0.8, 0.75, 0.65, 0.6, 0.78, 0.7, 0.05, 0.4, 0.4, 0.05, 0.5, 0.1, 0.1]
    actual = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

    gini_predictions = gini(actual, predictions)
    gini_max = gini(actual, actual)
    ngini = gini_normalized(actual, predictions)
    print('Gini: %.3f, Max. Gini: %.3f, Normalized Gini: %.3f' % (gini_predictions, gini_max, ngini))

    data = zip(actual, predictions)
    sorted_data = sorted(data, key=lambda d: d[1])
    sorted_actual = [d[0] for d in sorted_data]
    print('Sorted Actual Values', sorted_actual)