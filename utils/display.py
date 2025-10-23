import functools as ft

import numpy as np


def display_hist(x, num_bins=8, zeros_as_blank=False):
    """Returns a histogram as a unicode text string, e.g. '▁▂▄█▆▃▁▁'

    Inspired by the `precis` function from the Statistical Rethinking R package
    by Richard McElreath. This function will calculate a histogram and then
    returns a string displaying the histogram in unicode characters. It uses the
    LOWER BLOCK group like "2584 ▄ LOWER HALF BLOCK".

    After I published this I was alerted to the correct term for this type of text
    plot: spark lines. There is a python package by @RedKrieg that is much more
    robust for turning a sequence into a spark line called pysparklines. And the
    original(?) terminal package form @holman called spark:
    * pysparklines: https://github.com/RedKrieg/pysparklines
    * spark: https://github.com/holman/spark

    Parameters
    ----------
    x : numpy.array
        The vector of values to compute the histogram for
    num_bins : int or list of float
        The number of characters to print out. Can pass custom bin edges to
        `np.histogram` as well.
    zeros_as_blank : bool
        Should buckets with 0 observations be a blank space, False would still
        show a one eight block if there are no observations.

    Returns
    -------
    unicode_str : str
        The histogram str to be displayed

    Examples
    --------
    >>> display_hist(np.random.uniform(size=1000))
    '▇▇▇▇▇▆▇█'
    >>> display_hist(np.random.normal(size=1000))
    '▁▂▄█▆▃▁▁'
    >>> display_hist(np.abs(np.random.normal(size=1000)))
    '█▇▅▃▂▁▁▁'
    >>> display_hist(np.power(np.random.normal(size=1000), 2))
    '█▂▁▁▁▁▁▁'
    >>> display_hist(np.hstack([np.repeat(0, 900), np.repeat(10, 100)]), zeros_as_blank=True)
    '█      ▁'
    >>> display_hist(np.hstack([np.repeat(0, 900), np.repeat(10, 100)]))
    '█▁▁▁▁▁▁▁'
    >>> display_hist(np.hstack([np.random.normal(size = 1000),
                                np.random.normal(loc = 3, scale = 0.5, size = 1000)]),
                     num_bins = 16)
    '▁▁▂▂▃▅▄▄▂▁▃▆█▄▁▁'

    References
    ----------
    The unicode code charts: https://www.unicode.org/Public/UCD/latest/charts/CodeCharts.pdf
    """

    ## Get bin counts as a pct of total obs
    hist_counts, bin_edges = np.histogram(x, bins=num_bins)
    x_total = x.shape[0]
    pct_counts = hist_counts / x_total
    ## scale the percentages by the max pct and 0, then convert to the index
    ## of the appropriate unicode string in unicode_list
    max_pct = np.max(pct_counts)
    bin_labels = np.floor(pct_counts * (8 - 1) / max_pct).astype("int")
    ## adjust zeros to blank space index
    if zeros_as_blank:
        zero_ind = pct_counts == 0.0
        bin_labels[zero_ind] = 8

    unicode_list = ["\u2581", "\u2582", "\u2583", "\u2584", "\u2585", "\u2586", "\u2587", "\u2588", " "]
    unicode_labels = [unicode_list[label] for label in bin_labels]
    unicode_str = ft.reduce(lambda x, y: x + y, unicode_labels)
    return unicode_str
