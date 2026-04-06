import unittest
from cube import cube
class TestCube(unittest.TestCase):
    def test_cube(self):
        self.assertEqual(cube(0), 0)
        self.assertEqual(cube(1), 1)
        self.assertEqual(cube(-2), -8)
        self.assertEqual(cube(3.5), 42.875)

if __name__ == '__main__':
    unittest.main()