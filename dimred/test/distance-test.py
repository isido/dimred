""" Test cases for module distance"""

import unittest
import distance
import numpy, numpy.random

class DEuclideanTest(unittest.TestCase):

    def test_single_vector(self):
        """Distance from a random vector to the same random vector is zero"""
        vec = numpy.random.randn(1,50)[0,:]
        assert (distance.d_euclidean(vec, vec) == numpy.zeros([1,50])).all()

class DistanceMatrixTest(unittest.TestCase):
    def setUp(self):
        self.z = numpy.zeros([5,5])
        self.o = numpy.ones([5,5])


    def test_zero_matrix(self):
        """Distance matrix for zero matrix is zero"""
        assert (distance.distance_matrix(self.z) == self.z).all(), "Distance matrix for zero matrix"

    def test_unit_matrix(self):
        """Distance matrix for unit matrix is zero"""
        assert (distance.distance_matrix(self.o) == self.z).all(), "Distance matrix for unit matrix"

    def test_one_vector_matrix(self):
        """Distance matrix for matrix with one repeated random vector is zero"""

        vec = numpy.random.randn(1,50)
        mat = numpy.zeros([50,50])
        for row in range(50):
            mat[row,:] = vec

        assert (distance.distance_matrix(mat) == numpy.zeros([50,50])).all(), "Distance matrix for matrix composed from one random matrix"

    def test_two_points(self):
        # two points
        X = numpy.array([[0,0],[-1,0]])
        D = distance.distance_matrix(X)

        assert D.shape[0] == D.shape[1], "Distance matrix is square"

        assert D[0,0] == 0, "(0,0) - (0,0)"
        assert D[1,0] == 1, "(-1,0) - (0,0)"
        assert D[0,1] == 1, "(0,0) - (-1,0)"
        assert D[1,1] == 0, "(1,1) - (1,1)"
        

    def test_five_points(self):

        # five points
        X = numpy.array([[1,1], [2,2], [4,4], [3,5], [0,0], [2,0]])

        D = distance.distance_matrix(X)

        assert D.shape[0] == D.shape[1], "Distance matrix is square"

        for i in range(D.shape[0]):
            assert D[i,i] == 0, "Distance to vector itself is zero"

        # check precalculated distances
        assert D[4,5] == 2, "Distance between (0,0) and (2,0) is 2"
        assert D[5,4] == 2, "d (2,0) - (0,0) == 2"

    def test_random(self):

        i = numpy.random.randint(100)
        j = numpy.random.randint(100)

        a = numpy.random.random([i,j])
        d = distance.distance_matrix(a)

        assert d.shape[0] == d.shape[1], "Distance matrix is square"

        for i in range(d.shape[0]):
            assert d[i,i] == 0, "d(i,i) == 0"

        
class RankMatrixTest(unittest.TestCase):

    def test_three_points(self):
        a = numpy.array([[0,0], [1,1], [4,4]])
        d = distance.distance_matrix(a)
        r = distance.rank_matrix(d)

        assert r.shape[0] == r.shape[1], "Rank matrix is square"

        # rank(i,i) is zero
        for i in range(r.shape[0]):
            assert r[i,i] == 0, "Rank (i,i) is zero"

        assert r[0,1] == 1, "r(0,1)"
        assert r[0,2] == 2, "r(0,2)"
        assert r[1,0] == 1, "r(1,0)"
        assert r[1,2] == 2, "r(1,2)"
        assert r[2,0] == 2, "r(2,0)"
        assert r[2,1] == 1, "r(2,1)"

    def test_random(self):
        i = numpy.random.randint(100)
        j = numpy.random.randint(100)
        a = numpy.random.random([i,j])
        d = distance.distance_matrix(a)
        r = distance.rank_matrix(d)

        assert r.shape[0] == r.shape[1], "Rank matrix is square"

        for i in range(r.shape[0]):
            assert r[i,i] == 0, "Rank (i,i) is zero"

if __name__ == "__main__":
    unittest.main()
