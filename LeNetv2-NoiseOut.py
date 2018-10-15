##trying to prune LeNet 300-100 model.

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
X_train, y_train           = mnist.train.images, mnist.train.labels
X_validation, y_validation = mnist.validation.images, mnist.validation.labels
X_test, y_test             = mnist.test.images, mnist.test.labels

assert(len(X_train) == len(y_train))
assert(len(X_validation) == len(y_validation))
assert(len(X_test) == len(y_test))

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_validation)))
print("Test Set:       {} samples".format(len(X_test)))


import numpy as np

import random

import matplotlib.pyplot as plt

from scipy import stats


from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)


import tensorflow as tf

EPOCHS = 10
BATCH_SIZE = 128

from tensorflow.contrib.layers import flatten
from tensorflow.contrib.metrics import streaming_pearson_correlation

print('hello')


n_hidden_1 = 10  # 1st layer num features
n_hidden_2 = 100  # 2nd layer num features
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)
learning_rate_ini = 0.001

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

def get_correlation_matrix(data_matrix,NUM_OF_FEATURES):
    
    ##Create Matrix to store correlations of neuron output
    matrix_of_correlations_variable = [[0.0 for x in range(NUM_OF_FEATURES)] for y in range(NUM_OF_FEATURES)]

    for i in range( NUM_OF_FEATURES):
        for j in range(i+1,NUM_OF_FEATURES):
            _, matrix3 = streaming_pearson_correlation(data_matrix[:,i],data_matrix[:,j])
            matrix_of_correlations_variable[i][j]=matrix3
    
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    return matrix_of_correlations_variable

def get_matrix_arg_max_indices(tensor_matrix_correlation,NUM_OF_FEATURES):

    x = tf.reshape(tensor_matrix_correlation,  [-1])
    max_index = tf.argmax(x)
    col_indices = max_index // NUM_OF_FEATURES
    row_indices = max_index % NUM_OF_FEATURES
    final_indices = tf.stack([col_indices,row_indices])
    return final_indices

def update_bias(bias_matrix,index_to_remove,weight_matrix_prior,intercept):
    initial_bias_matrix = sess.run(bias_matrix)
    initial_weight_matrix = sess.run(weight_matrix_prior)
    intercept_1 = sess.run(intercept)
    initial_bias_matrix = initial_bias_matrix + intercept_1*weight_matrix_prior[index_to_remove,:]
    initial_bias_matrix1 = tf.convert_to_tensor(initial_bias_matrix)
    return initial_bias_matrix

# def remove_neuron_input(x, final_indices,slope):
#     initial_weight_matrix = sess.run(x)
#     # test_array = sess.run(final_indices)
#     test_array = final_indices
#     index_to_remove = test_array[1]
#     index_to_update = test_array[0]
#     initial_weight_matrix[:,index_to_update] = slope*initial_weight_matrix[:,index_to_remove] + initial_weight_matrix[:,index_to_update] 
#     initial_weight_matrix[:,index_to_remove] = 0
#     initial_weight_matrix1 = tf.convert_to_tensor(initial_weight_matrix)
#     return initial_weight_matrix1

def remove_neuron_input(x, index_to_remove_1,index_to_update_1,slope):
    initial_weight_matrix = x
    initial_weight_matrix_1 = initial_weight_matrix

    one = initial_weight_matrix[:,index_to_update_1].assign(initial_weight_matrix[:,index_to_update_1]  + tf.cast(slope,tf.float32)*initial_weight_matrix[:,index_to_remove_1]) 

    shape_test = tf.shape(initial_weight_matrix[:,index_to_remove_1])
    a = tf.Variable(tf.zeros(shape_test,tf.float32))
    
    two = initial_weight_matrix[:,index_to_remove_1].assign(a)


    return two

def remove_neuron_output(x,index_to_remove,index_to_update,slope):
    initial_weight_matrix = sess.run(x)
    initial_weight_matrix[index_to_update,:] = slope*initial_weight_matrix[index_to_remove,:] + initial_weight_matrix[index_to_update,:] 
    initial_weight_matrix[index_to_remove,:] = 0
    return initial_weight_matrix 

def model(_X, _W, _biases):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _W['fc1']), _biases['fc1']))  # Hidden layer with RELU activation
    tf.nn.dropout(layer_1, 0.5)
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _W['fc2']), _biases['fc2']))  # Hidden layer with RELU activation
    tf.nn.dropout(layer_2, 0.5)
    return tf.matmul(layer_2, _W['out']) + _biases['out']

def model_prune(_X, _W, _biases):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _W['fc1']), _biases['fc1']))  # Hidden layer with RELU activation
    matrix_of_correlations_variable = get_correlation_matrix(layer_1,n_hidden_1)
    tensor_matrix_correlation = tf.convert_to_tensor(matrix_of_correlations_variable)
    final_indices = get_matrix_arg_max_indices(tensor_matrix_correlation,n_hidden_1)
    slope, intercept, r_value, p_value, std_err= tf.py_func(stats.linregress,[layer_1[:,final_indices[0]],layer_1[:,final_indices[1]]],[tf.float64,tf.float64,tf.float64,tf.float64,tf.float64])

    #index to remove is always [1]
    initial_bias_matrix_1 = update_bias(_biases['fc1'],final_indices[1],_W['fc1'],intercept)
    _biases['fc1'].assign(initial_bias_matrix_1, use_locking=False)

    initial_weight_matrix_1 = remove_neuron_input(_W['fc1'],final_indices[1],final_indices[0],slope)
    _W['fc1'].assign(initial_weight_matrix_1, use_locking=False)



    
    tf.nn.dropout(layer_1, 0.5)

    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _W['fc2']), _biases['fc2']))  # Hidden layer with RELU activation
    tf.nn.dropout(layer_2, 0.5)
    return tf.matmul(layer_2, _W['out']) + _biases['out']

# Store layers weight & bias
W = {
    'fc1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=0.01)),
    'fc2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=0.01)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], stddev=0.01))
}

W_prune = W

biases = {
    'fc1': tf.Variable(tf.random_normal([n_hidden_1], stddev=0.01)),
    'fc2': tf.Variable(tf.random_normal([n_hidden_2], stddev=0.01)),
    'out': tf.Variable(tf.random_normal([n_classes], stddev=0.01))
}

pred = model(x, W, biases)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))  # Softmax loss
cost = loss

optimizer = tf.train.AdamOptimizer(
        learning_rate_ini, beta1=0.9, beta2=0.999,
        epsilon=1e-08, use_locking=False).minimize(cost)

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# Calculate accuracy
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples



with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()

    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            # print(i,offset,LeNet(batch_x))
            # test_output = LeNet(X_train[1:128])
            # print(test_output.eval())
            _,loss_val=sess.run([optimizer,loss], feed_dict={x: batch_x, y: batch_y})
            #if cost(W)<=threshold. Threshold selected as 0.02
            if loss_val<=0.20:
                print(loss_val)
                print(sess.run(W['fc1']))

                model_prune(batch_x, W, biases)
                print(sess.run(W['fc1']))
                # sess.run(tf.global_variables_initializer())
                # sess.run(tf.local_variables_initializer())


            
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
