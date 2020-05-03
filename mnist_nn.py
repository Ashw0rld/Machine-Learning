# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 04:08:35 2020

@author: KIIT
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

no_node_hl1 = 500
no_node_hl2 = 500
no_node_hl3 = 500

no_classes = 10

batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([784, no_node_hl1])), 
                      'biases': tf.Variable(tf.random_normal([no_node_hl1]))}
    
    hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([no_node_hl1, no_node_hl2])),
                        'biases': tf.Variable(tf.random_normal([no_node_hl2]))}
    
    hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([no_node_hl2, no_node_hl3])),
                        'biases': tf.Variable(tf.random_normal([no_node_hl3]))}
    
    output_layer = {'weights': tf.Variable(tf.random_normal([no_node_hl3, no_classes])),
                        'biases': tf.Variable(tf.random_normal([no_classes]))}
    
    l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
    l1 = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
    l2 = tf.nn.relu(l2)
    
    l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['biases'])
    l3 = tf.nn.relu(l3)
    
    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
    
    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 10
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'cpmpleted out of', hm_epochs, 'loss:', epoch_loss)
            
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
        
train_neural_network(x)
                
