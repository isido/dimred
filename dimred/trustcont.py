#
#    Dimensionality Reduction Tools
#    Copyright (C) 2010 Ilja Sidoroff
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#    trustcont.py
#
#    Trustworthiness and Continuity error measures for reduced datasets
#
#    See e. g. Jarkko Venna, `Dimensionality Reduction for Visual
#    Exploration of Similarity Structures', Helsinki University of
#    Technology, Dissertations in Computer and Information Science, Espoo
#    2007

import distance

def trustworthiness(orig, proj, ks):
    """Calculate a trustworthiness values for dataset.

    orig
      matrix containing the data in the original space

    proj
      matrix containing the data in the projected space

    ks range indicating neighbourhood(s) for which
      trustworthiness is calculated.

    Return list of trustworthiness values

    """

    dd_orig = distance.distance_matrix(orig)
    dd_proj = distance.distance_matrix(proj)
    nn_orig = dd_orig.argsort()
    nn_proj = dd_proj.argsort()

    ranks_orig = distance.rank_matrix(dd_orig)

    trust = []
    for k in ks:
        moved = []
        for i in range(orig.shape[0]):
            moved.append(moved_in(nn_orig, nn_proj, i, k))

        trust.append(trustcont_sum(moved, ranks_orig, k))

    return trust

def continuity(orig, proj, ks):
    """Calculate a continuity values for dataset

    orig
      matrix containing the data in the original space

    proj
      matrix containing the data in the projected space

    ks range indicating neighbourhood(s) for which continuity
      is calculated.

    Return a list of continuity values
    """

    dd_orig = distance.distance_matrix(orig)
    dd_proj = distance.distance_matrix(proj)
    nn_orig = dd_orig.argsort()
    nn_proj = dd_proj.argsort()

    ranks_proj = distance.rank_matrix(dd_proj)

    cont = []
    for k in ks:
        moved = []
        for i in range(orig.shape[0]):
            moved.append(moved_out(nn_orig, nn_proj, i, k))

        cont.append(trustcont_sum(moved, ranks_proj, k))

    return cont

def moved_out(nn_orig, nn_proj, i, k):
    """Determine points that were neighbours in the original space,
    but are not neighbours in the projection space.

    nn_orig
      neighbourhood matrix for original data

    nn_proj
      neighbourhood matrix for projection data

    i
      index of the point considered

    k
      size of the neighbourhood considered

    Return a list of indices for 'moved out' values 
    """

    oo = list(nn_orig[i, 1:k+1])
    pp = list(nn_proj[i, 1:k+1])

    for j in pp:
        if (j in pp) and (j in oo):
            oo.remove(j)

    return oo

def moved_in(nn_orig, nn_proj, i, k):
    """Determine points that are neighbours in the projection space,
    but were not neighbours in the original space.

    nn_orig
      neighbourhood matrix for original data

    nn_proj
      neighbourhood matrix for projection data

    i
      index of the point considered

    k
      size of the neighbourhood considered

    Return a list of indices for points which are 'moved in' to point i
    """

    pp = list(nn_proj[i, 1:k+1])
    oo = list(nn_orig[i, 1:k+1])

    for j in oo:
        if (j in oo) and (j in pp):
            pp.remove(j)

    return pp


def scaling_term(k, n):
    """Term that scales measure between zero and one

    k  size of the neighbourhood
    n  number of datapoints
    """
    if k < (n / 2.0):
        return 2.0 / ((n*k)*(2*n - 3*k - 1))
    else:
        return 2.0 / (n * (n - k) * (n - k - 1))


def trustcont_sum(moved, ranks, k):
    """Calculate sum used in trustworthiness or continuity calculation.

    moved

       List of lists of indices for those datapoints that have either
       moved away in (Continuity) or moved in (Trustworthiness)
       projection

    ranks

       Rank matrix of data set. For trustworthiness, ranking is in the
       original space, for continuity, ranking is in the projected
       space.

    k
       size of the neighbournood
    """

    n = ranks.shape[0]
    s = 0

    # todo: weavefy this for speed
    for i in range(n):
        for j in moved[i]:
            s = s + (ranks[i, j] - k)

    a = scaling_term(k, n)

    return 1 - a * s
