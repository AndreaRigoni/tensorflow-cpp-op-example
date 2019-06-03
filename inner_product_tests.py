#!/usr/bin/env python3
"""
Tests for the inner product Tensorflow operation.

.. moduleauthor:: David Stutz
"""

import unittest
import numpy as np
import tensorflow as tf
import _inner_product_grad
inner_product_module = tf.load_op_library('libinner_product.so')

class InnerProductOpTest(unittest.TestCase):
    # def test_raisesExceptionWithIncompatibleDimensions(self):
        # with self.assertRaises(ValueError):
        #     inner_product_module.inner_product([1, 2], [[1, 2], [3, 4]])
        # with self.assertRaises(ValueError):
        #     self.assertRaises(inner_product_module.inner_product([1, 2], [1, 2, 3, 4]), ValueError)
        #     with self.assertRaises(ValueError):
        #         self.assertRaises(inner_product_module.inner_product([1, 2, 3], [[1, 2], [3, 4]]), ValueError)
            
    def test_innerProductHardCoded(self):
        result = inner_product_module.inner_product([[1], [2]], [[1, 2], [3, 4]])
        self.assertEqual(result.shape[0], 2)
        self.assertEqual(result[0].numpy(), 5)
        self.assertEqual(result[1].numpy(), 11)
    
    def test_innerProductGradientXHardCoded(self):
        x = tf.convert_to_tensor(np.asarray([1, 2]).astype(np.float32))
        W = tf.constant(np.asarray([[1, 2], [3, 4]]).astype(np.float32))
        with tf.GradientTape(persistent=True) as g:
            g.watch(x)            
            Wx_tf = tf.matmul(W, tf.reshape(x, [-1, 1]))
            Wx_inner_product = inner_product_module.inner_product(tf.reshape(x, [-1, 1]), W)                        
            
        gradient_tf = g.gradient(Wx_tf,x)
        gradient_inner_product = g.gradient(Wx_inner_product,x)        
        self.assertEqual(gradient_tf[0].numpy(), gradient_inner_product[0].numpy())
        self.assertEqual(gradient_tf[1].numpy(), gradient_inner_product[1].numpy())

    def test_innerProductGradientWHardCoded(self):
        x = tf.constant(np.asarray([1, 2]).astype(np.float32))        
        W = tf.convert_to_tensor(np.asarray([[1, 2], [3, 4]]).astype(np.float32))
        with tf.GradientTape(persistent=True) as g:
            g.watch(W)            
            Wx_tf = tf.matmul(W, tf.reshape(x, [-1, 1]))
            Wx_inner_product = inner_product_module.inner_product(tf.reshape(x, [-1, 1]), W)
            
        gradient_tf = g.gradient(Wx_tf, W)
        gradient_inner_product = g.gradient(Wx_inner_product, W)                    
        self.assertEqual(gradient_tf[0][0].numpy(), gradient_inner_product[0][0].numpy())
        self.assertEqual(gradient_tf[0][1].numpy(), gradient_inner_product[0][1].numpy())
        self.assertEqual(gradient_tf[1][0].numpy(), gradient_inner_product[1][0].numpy())
        self.assertEqual(gradient_tf[1][1].numpy(), gradient_inner_product[1][1].numpy())
    
    def test_innerProductRandom(self):
        n = 4
        m = 5
        
        for i in range(100):
            x_rand = np.random.randint(10, size = (n, 1))
            W_rand = np.random.randint(10, size = (m, n))
            result_rand = np.dot(W_rand, x_rand)
            
            result = inner_product_module.inner_product(x_rand, W_rand)
            np.testing.assert_array_equal(result, result_rand)
    
    def test_innerProductGradientXRandom(self):
        # with tf.Session('') as sess:
        n = 4
        m = 5                
        
        for i in range(100):
            x = tf.convert_to_tensor(np.random.randint(10, size = (n)), dtype=tf.float32)
            W = tf.convert_to_tensor(np.random.randint(10, size = (m, n)), dtype=tf.float32)
            with tf.GradientTape(persistent=True) as g:
                g.watch(x)            
                Wx_tf = tf.matmul(W, tf.reshape(x, [-1, 1]))
                Wx_inner_product = inner_product_module.inner_product(tf.reshape(x, [-1, 1]), W)

            gradient_tf = g.gradient(Wx_tf, x)
            gradient_inner_product = g.gradient(Wx_inner_product, x)
            np.testing.assert_array_equal(gradient_tf.numpy(), gradient_inner_product.numpy())
                

                
if __name__ == '__main__':
    unittest.main()