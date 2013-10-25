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
#    isomap.py: an implementation of Isomap-algorithm
#
#    See for instance http://isomap.stanford.edu
#
#    or read
#
#    A Global Geometric Framework for Nonlinear Dimensionality Reduction;
#    J.B. Tenenbaum, V. de Silva and J. C. Langford; Science 290(5500),
#    pages 2319--2323


import networkx as nx
import numpy

import distance as dd
import graph

def isomap(nn_graph, target_dim):
    """Isomap algorithm

    nn_graph      matrix containing nearest neighbourhood graph
    target_dim    target dimension for projection
    """

    # compute shortest distances, square them, and store into D

    # double center D

    # calculate eigen-decomposition D = ULUt

    # project data Y = I L U
