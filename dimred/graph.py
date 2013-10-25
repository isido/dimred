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
#    graph.py: graph-related functions
#

def nn_graph(x, k):
    """Create nearest neighbourhood graph from data, using
    number of neighbours

    """
    # create graph
    dd = dd.distance_matrix(x) 
    nn =  dd.argsort()[:,1:k+1]

    nn_graph = nx.Graph()

    nodes = range(nn.shape[0])
    nn_graph.add_nodes_from(nodes)
    for i in nodes:
        edges = [ (i, x) for x in nn[i, :] ]
        nn_graph.add_edges_from(edges)

    # weigh edges by distance
