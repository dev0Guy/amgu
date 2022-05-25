import unittest


class TestEnvironment(unittest.TestCase):
    def extract_information(self):
        self.assertEqual("x", "x")


if __name__ == "__main__":
    unittest.main()
