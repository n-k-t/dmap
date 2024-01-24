import unittest
from memory import Memory
from tensor import Tensor

class TestMemoryInit(unittest.TestCase):
    def test_memory_init(self):
        test = Memory(shape = [1])
        self.assertIsInstance(test, Memory, "The object is not of class 'Memory'.")
        test = Memory(shape = [121, 5, 8])
        self.assertIsInstance(test, Memory, "The object is not of class 'Memory'.")

class TestMemoryMethods(unittest.TestCase):
    def test_view_from_shape(self):
        test = Memory(shape = [1])
        self.assertEqual(test.view, [1])
        test = Memory(shape = [10, 10, 2, 3])
        self.assertEqual(test.view, [10, 10, 2, 3])
        test = Memory(shape = [5, 2, 1, 1, 1, 1, 1])
        self.assertEqual(test.view, [5, 2, 1, 1, 1, 1, 1])

    def test_stride_from_shape(self):
        test = Memory(shape = [1])
        self.assertEqual(test.stride, [1])
        test = Memory(shape = [10, 10, 2, 3])
        self.assertEqual(test.stride, [60, 6, 3, 1])
        test = Memory(shape = [5, 2, 1, 1, 1, 1, 1])
        self.assertEqual(test.stride, [2, 1, 1, 1, 1, 1, 1])
    
    def test_contiguity(self):
        test = Memory(shape = [1])
        self.assertTrue(test._contiguous)
        test = Memory(shape = [6, 3, 98, 2])
        self.assertTrue(test._contiguous)
        test = Memory(shape = [6, 3, 98, 2], safe_op = False, stride = [588, 196, 2, 1])
        self.assertTrue(test._contiguous)
        test = Memory(shape = [6, 3, 98, 2], safe_op = False, stride = [588, 2, 1, 196])
        self.assertFalse(test._contiguous)
        test = Memory(shape = [6, 3, 98, 2], safe_op = False, stride = [2, 3])
        self.assertFalse(test._contiguous)

class TestMemoryFailure(unittest.TestCase):
    def test_shape_zero_failure(self):
        with self.assertRaises(ValueError):
            Memory(shape = [0])
        with self.assertRaises(ValueError):
            Memory(shape = [0, 1, 10])
        with self.assertRaises(ValueError):
            Memory(shape = [1, 0, 10])
        with self.assertRaises(ValueError):
            Memory(shape = [1, 10, 0])

    def test_shape_negative_failure(self):
        with self.assertRaises(ValueError):
            Memory(shape = [-1])
        with self.assertRaises(ValueError):
            Memory(shape = [1, 1, -10])
        with self.assertRaises(ValueError):
            Memory(shape = [1, -1, 10])
        with self.assertRaises(ValueError):
            Memory(shape = [-1, 10, 1])

    def test_shape_non_integer_failure(self):
        with self.assertRaises(TypeError):
            Memory(shape = [1.])
        with self.assertRaises(TypeError):
            Memory(shape = [1.4, 3.1, 7.0])
        with self.assertRaises(TypeError):
            Memory(shape = ["a", "b", "c"])
        with self.assertRaises(TypeError):
            Memory(shape = [True, False])

    def test_memory_no_args_failure(self):
        with self.assertRaises(TypeError):
            Memory()

class TestTensorInit(unittest.TestCase):
    def test_tensor_creation(self):
        test = Tensor(shape = [1])
        self.assertIsInstance(test, Tensor, "The object is not of class 'Tensor'.")
        test = Tensor(shape = [121, 5, 8])
        self.assertIsInstance(test, Tensor, "The object is not of class 'Tensor'.")

class TestTensorFailure(unittest.TestCase):
    def test_tensor_bad_shape_failure(self):
        with self.assertRaises(ValueError):
            Tensor(shape = [-1])
        with self.assertRaises(ValueError):
            Tensor(shape = [1, 1, -10])
        with self.assertRaises(ValueError):
            Tensor(shape = [1, 1, 0])

    def test_tensor_wrong_type_shape_args_failure(self):
        with self.assertRaises(TypeError):
            Tensor(shape = [1.])
        with self.assertRaises(TypeError):
            Tensor(shape = [1.4, 3.1, 7.0])
        with self.assertRaises(TypeError):
            Tensor(shape = ["a", "b", "c"])
        with self.assertRaises(TypeError):
            Tensor(shape = [True, False])

    def test_tensor_no_args_failure(self):
        with self.assertRaises(TypeError):
            Tensor()

if __name__ == '__main__':
    unittest.main(verbosity = 2)