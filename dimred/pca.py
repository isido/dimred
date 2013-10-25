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
#    pca.py: an implementation of principal component analysis
#

import numpy.linalg

import distance as dd

def pca_cov(x, target_dim):
    """Calculate principal component projection for data using covariance
    matrix.

    x           matrix of original data
    target_dim  dimensionality of the projection


    Return projected data, sorted eigenvalues and eigenvectors for original
    data
    """

    # center data and calculate eigenvalues and -vectors for covariance matrix
    centered = dd.center(x)
    cov = numpy.cov(centered)
    (eigenvalues, eigenvectors) = numpy.linalg.eig(cov)

    # sort eigenvalues and vectors
    ind = eigenvalues.argsort()
    ind = ind[::-1]
    eigenvalues = eigenvalues[ind]
    eigenvectors = eigenvectors[:, ind]

    # compute projection
    project = eigenvectors[:,:target_dim]
    y = dot(transpose(eigenvectors), transpose(x))

    return y, eigenvalues, eigenvectors
