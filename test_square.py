import unittest
from square import square
class TestSquare(unittest.TestCase):
    def test_square(self):
        self.assertEqual(square(0), 0)
        self.assertEqual(square(1), 1)
        self.assertEqual(square(-2), 4)
        self.assertEqual(square(3.5), 12.25)

if __name__ == '__main__':
    unittest.main()