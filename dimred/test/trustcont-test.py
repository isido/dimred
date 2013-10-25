"""Test cases for module trustcont"""

import unittest

import numpy

import distance
import trustcont

eps = 0.00001

class ScalingTermTest(unittest.TestCase):

    def test_division_by_zero(self):
        try:
            trustcont.scaling_term(1,0)
        except ZeroDivisionError:
            pass
        else:
            fail("excepted a ZeroDivisionError")

        try:
            trustcont.scaling_term(0,1)
        except ZeroDivisionError:
            pass
        else:
            fail("excepted a ZeroDivisionError")


    def test_illegal_values(self):
        try:
            trustcont.scaling_term(1,1)
        except ZeroDivisionError:
            pass
        else:
            fail("excepted a ZeroDivisionError")

        try:
            trustcont.scaling_term(10,20)
        except:
            # should probably emit a warning or do something
            pass

        try:
            trustcont.scaling_term(-10,20)
        except:
            # should probably emit a warning
            pass
        
        try:
            trustcont.scaling_term(10,-20)
        except:
            # should probably emit a warning
            pass
           
    def test_precalculated_values(self):

        values = [(1, 10, 0.0125),
                  (4, 10, 0.0071428571428571426),
                  (5, 10, 0.01)
                  ]

        for (k, n, res) in values:
            r = trustcont.scaling_term(k, n)
            s = "( " + str(k) + ", " + str(n) + ") => " + str(r)
            assert r == res, s



class MovedInOutTest(unittest.TestCase):

    def setUp(self):
        """
        original:   projection:

        2
        0 1         0     2 1
        
        origo = (0)

        """
        self.orig = numpy.array([[0,0],   
                                 [2,0],   
                                 [0,1]])  

        self.proj = numpy.array([[0,0],
                                 [5,0],
                                 [4,0]])
        
        self.dd_orig = distance.distance_matrix(self.orig)
        self.dd_proj = distance.distance_matrix(self.proj)

        self.nn_orig = self.dd_orig.argsort()
        self.nn_proj = self.dd_proj.argsort()

    def test_random(self):
        items = 10
        dim = 10
        orig = numpy.random.randn(items,dim)
        proj = numpy.random.randn(items,dim)

        dd_orig = distance.distance_matrix(orig)
        dd_proj = distance.distance_matrix(proj)

        nn_orig = dd_orig.argsort()
        nn_proj = dd_proj.argsort()

        for i in range(items):
            moved_in = trustcont.moved_in(nn_orig, nn_proj, i, i)
            assert (not i in moved_in), "Point itself cannot be in its'\
             own neighbour"
            moved_out = trustcont.moved_out(nn_orig, nn_proj, i, i)
            assert (not i in moved_out), "Point itself cannot be in its'\
             own neighbour"

    def test_moved_ins_0(self):
        # point 0, k=1 => []
        moved_in = trustcont.moved_in(self.nn_orig, self.nn_proj, 0, 1)
        assert moved_in == [], "Result: " + str(moved_in)

    def test_moved_ins_1(self):
        # point 1, k=1 => [2]
        moved_in = trustcont.moved_in(self.nn_orig, self.nn_proj, 1, 1)
        assert moved_in == [2], "Result: " + str(moved_in)

    def test_moved_ins_2(self):
        # point 2, k=1 => [1]
        moved_in = trustcont.moved_in(self.nn_orig, self.nn_proj, 2, 1)
        assert moved_in == [1], "Result: " + str(moved_in)

    def test_moved_ins_k2(self):
        # for neigbourhood of 2, there shouldn't be moved ins
        for i in range(len(self.orig)):
            moved_in = trustcont.moved_in(self.nn_orig, self.nn_proj, i, 2)
            assert moved_in == [], "Result: " + str(i) + " " + str(moved_in)

    def test_moved_ins_k3(self):
        # for neighbourhood of 3, there shouldn't be moved ins
        for i in range(len(self.orig)):
            moved_in = trustcont.moved_in(self.nn_orig, self.nn_proj, i, 3)
            assert moved_in == [], "Result: " + str(i) + " " + str(moved_in)

    def test_moved_outs_0(self):
        # point 0, k=1 => []
        moved_out = trustcont.moved_out(self.nn_orig, self.nn_proj, 0, 1)
        assert moved_out == [], "Result: " + str(moved_out)

    def test_moved_outs_1(self):
        # point 1, k=1 => [0]
        moved_out = trustcont.moved_out(self.nn_orig, self.nn_proj, 1, 1)
        assert moved_out == [0], "Result: " + str(moved_out)

    def test_moved_outs_2(self):
        # point 2, k=1 => [0]
        moved_out = trustcont.moved_out(self.nn_orig, self.nn_proj, 2, 1)
        assert moved_out == [0], "Result: " + str(moved_out)

    def test_moved_outs_k2(self):
        # for neighbourhood of 2, there shouldn't be moved outs
        for i in range(len(self.orig)):
            moved_out = trustcont.moved_out(self.nn_orig, self.nn_proj, i, 2)
            assert moved_out == [], "Result: " + str(i) + " "+ str(moved_out)

    def test_moved_outs_k3(self):
        # for neighbourhood of 3, there shouldn't be moved outs
        for i in range(len(self.orig)):
            moved_out = trustcont.moved_out(self.nn_orig, self.nn_proj, i, 3)
            assert moved_out == [], "Result: " + str(i) + " " + str(moved_out)


class TrustContSumTest(unittest.TestCase):

    def test_sum_k1_moved_in(self):
        """
        Test precalculated sums for neighbourhood of 1
        """
        # moved ins = trustworthiness
        moved_ins = [[], [2], [1]]
        # ranks in original space
        ranks = numpy.array([[0,2,1], [1,0,2], [1,2,0]])

        expected = 0.33333333333
        calc = trustcont.trustcont_sum(moved_ins, ranks, 1)
        
        assert abs(expected - calc) < eps,\
               "Expected " + str(expected) + ", got " + str(calc)


    def test_sum_k1_moved_out(self):
        """
        Test precalculated sums for neighbourhood of 1
        """
        # moved outs = continuity
        moved_outs = [[], [0], [0]]
        # ranks in projection space
        ranks = numpy.array([[0,2,1], [2,0,1], [2,1,0]])

        expected = 0.3333333
        calc = trustcont.trustcont_sum(moved_outs, ranks, 1)

        assert abs(expected - calc) < eps, \
               "Expected " + str(expected) + ", got " + str(calc)
        

class TrustContTest(unittest.TestCase):

    def setUp(self):
        self.orig = numpy.array([[0,0], [2,0], [0,1]])
        self.proj = numpy.array([[0,0], [5,0], [4,0]])


    def test_trust(self):
        ks = [1]
        expected = [0.33333]

        res = trustcont.trustworthiness(self.orig, self.proj, ks)

        for (val, e) in zip(res, expected):
            assert abs(val - e) < eps, "Expected " + str(e) + ", got "+\
                   str(val)

            
    def test_cont(self):
        ks = [1]
        expected = [0.33333]

        res = trustcont.continuity(self.orig, self.proj, ks)

        for (val, e) in zip(res, expected):
            assert abs(val - e) < eps, "Expected " + str(e) + ", got "+\
                   str(val)
            
