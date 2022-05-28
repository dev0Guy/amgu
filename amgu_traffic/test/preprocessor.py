import unittest
from amgu_traffic import preprocessor
import numpy as np

s0 = np.zeros((3, 1, 32, 40))

class PreprocessorTest(unittest.TestCase):
    def testLaneQeueueLength(self):
        a = np.array([0]*32)
        b = preprocessor.LaneQeueueLength(s0,2)[0]
        self.assertEqual(np.all(a==b),True)
        s0[2] += 1
        a = np.array([40]*32)
        b = preprocessor.LaneQeueueLength(s0,2)[0]
        self.assertEqual(np.all(a==b),True)

    def test_vanila(self):
        a = s0
        b = preprocessor.Vanila(s0)
        self.assertEqual(np.all(a==b),True)

if __name__ == "__main__":
    unittest.main()
