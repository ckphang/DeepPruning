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
from scipy.stats.stats import pearsonr 

from sklearn.utils import shuffle

np.set_printoptions(threshold=10)

X_train, y_train = shuffle(X_train, y_train)


import tensorflow as tf

EPOCHS = 10
BATCH_SIZE = 128

from tensorflow.contrib.layers import flatten
from tensorflow.contrib.metrics import streaming_pearson_correlation

print('hello')


n_hidden_1 = 50  # 1st layer num features
n_hidden_2 = 50  # 2nd layer num features
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)
learning_rate_ini = 0.001

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

def get_correlation_matrix(data_matrix1,NUM_OF_FEATURES):
    
    ##Create Matrix to store correlations of neuron output
    data_matrix2 = sess.run([data_matrix1])
    data_matrix = data_matrix2[0]

    matrix_of_correlations_variable = [[0.0 for x in range(NUM_OF_FEATURES)] for y in range(NUM_OF_FEATURES)]

    for i in range( NUM_OF_FEATURES):
        if np.sum(data_matrix[:,i])==0:
            pass
        for j in range(i+1,NUM_OF_FEATURES):
            if np.sum(data_matrix[:,j])==0:
                pass
            matrix3,_ = pearsonr(data_matrix[:,i],data_matrix[:,j])
            matrix_of_correlations_variable[i][j]=matrix3

    return matrix_of_correlations_variable

# def get_correlation_matrix_original(data_matrix1,NUM_OF_FEATURES):
    
#     ##Create Matrix to store correlations of neuron output
#     data_matrix = sess.run([data_matrix1])
#     data_matrix = np.array(data_matrix)

#     matrix_of_correlations_variable = [[0.0 for x in range(NUM_OF_FEATURES)] for y in range(NUM_OF_FEATURES)]

#     for i in range( NUM_OF_FEATURES):
#         if np.sum(data_matrix[:,i])==0:
#             pass
#         for j in range(i+1,NUM_OF_FEATURES):
#             if np.sum(data_matrix[:,j])==0:
#                 pass
#             matrix3,_ = pearsonr(data_matrix[:,i],data_matrix[:,j])
#             matrix_of_correlations_variable[i][j]=matrix3

#     return matrix_of_correlations_variable

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
    index_to_remove_1 = sess.run(index_to_remove) 
    initial_bias_matrix1 = initial_bias_matrix + intercept_1*initial_weight_matrix[index_to_remove_1,:]
    initial_bias_matrix2 = tf.convert_to_tensor(initial_bias_matrix1)
    return initial_bias_matrix2

def remove_neuron_input(x, index_to_remove_1,index_to_update_1,slope):
    initial_weight_matrix = sess.run(x)
    # test_array = sess.run(final_indices)
    index_to_remove_2 = sess.run(index_to_remove_1)
    index_to_update_2 = sess.run(index_to_update_1)
    slope2 = sess.run(slope)

    initial_weight_matrix[:,index_to_update_2] = slope2*initial_weight_matrix[:,index_to_remove_2] + initial_weight_matrix[:,index_to_update_2] 
    initial_weight_matrix[:,index_to_remove_2] = 0
    initial_weight_matrix1 = tf.convert_to_tensor(initial_weight_matrix)
    return initial_weight_matrix1



# def remove_neuron_input(x, index_to_remove_1,index_to_update_1,slope):
#     initial_weight_matrix = x
#     initial_weight_matrix_1 = initial_weight_matrix

#     one = initial_weight_matrix[:,index_to_update_1].assign(initial_weight_matrix[:,index_to_update_1]  + tf.cast(slope,tf.float32)*initial_weight_matrix[:,index_to_remove_1]) 

#     shape_test = tf.shape(initial_weight_matrix[:,index_to_remove_1])
#     a = tf.Variable(tf.zeros(shape_test,tf.float32))
    
#     two = initial_weight_matrix[:,index_to_remove_1].assign(a)

#     return two

def remove_neuron_output(x,index_to_remove,index_to_update,slope):
    initial_weight_matrix = sess.run(x)
    index_to_remove_1 = sess.run(index_to_remove)
    index_to_update_1 = sess.run(index_to_update)
    slope1 = sess.run(slope)
    initial_weight_matrix[index_to_update_1,:] = slope1*initial_weight_matrix[index_to_remove_1,:] + initial_weight_matrix[index_to_update_1,:] 
    initial_weight_matrix[index_to_remove_1,:] = 0
    initial_weight_matrix1 = tf.convert_to_tensor(initial_weight_matrix)
    return initial_weight_matrix1 

# def remove_neuron_output(x,index_to_remove,index_to_update,slope):
#     initial_weight_matrix = x
#     one = initial_weight_matrix[index_to_update,:].assign(tf.cast(slope,tf.float32)*initial_weight_matrix[index_to_remove,:] + initial_weight_matrix[index_to_update,:])

#     shape_test = tf.shape(initial_weight_matrix[index_to_remove,:])
#     a = tf.Variable(tf.zeros(shape_test,tf.float32)) 
#     two = initial_weight_matrix[index_to_remove,:].assign(a) 
#     return two

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

    # #index to remove is always [1]
    initial_bias_matrix_1 = update_bias(_biases['fc1'],final_indices[1],_W['fc1'],intercept)
    assign_bias_op1 =_biases['fc1'].assign(initial_bias_matrix_1, use_locking=False)
    sess.run(assign_bias_op1)
    initial_weight_matrix_1 = remove_neuron_input(_W['fc1'],final_indices[1],final_indices[0],slope)
    assign_weight1_op = _W['fc1'].assign(initial_weight_matrix_1, use_locking=False)
    sess.run(assign_weight1_op)
    initial_weight_matrix_1_out = remove_neuron_output(_W['fc2'],final_indices[1],final_indices[0],slope)
    assign_weight1out_op = _W['fc2'].assign(initial_weight_matrix_1_out, use_locking=False)
    sess.run(assign_weight1out_op)
    
    # # tf.nn.dropout(layer_1, 0.5)

    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _W['fc2']), _biases['fc2']))  # Hidden layer with RELU activation
    matrix_of_correlations_variable_2 = get_correlation_matrix(layer_2,n_hidden_2)
    tensor_matrix_correlation_2 = tf.convert_to_tensor(matrix_of_correlations_variable_2)
    final_indices_2 = get_matrix_arg_max_indices(tensor_matrix_correlation_2,n_hidden_2)
    slope2,intercept2,r_value2,p_value2,std_err2 = tf.py_func(stats.linregress,[layer_2[:,final_indices_2[0]],layer_2[:,final_indices_2[1]]],[tf.float64,tf.float64,tf.float64,tf.float64,tf.float64])

    initial_bias_matrix_2 = update_bias(_biases['fc2'],final_indices_2[1],_W['fc2'],intercept2)
    assign_bias_op2 = _biases['fc2'].assign(initial_bias_matrix_2,use_locking = False)
    sess.run(assign_bias_op2)

    initial_weight_matrix_2 = remove_neuron_input(_W['fc2'],final_indices_2[1],final_indices_2[0],slope2)
    assign_weight2_op=_W['fc2'].assign(initial_weight_matrix_2,use_locking=False)
    sess.run(assign_weight2_op)

    initial_weight_matrix_2_out = remove_neuron_output(_W['out'],final_indices[1],final_indices[0],slope)
    assign_weight2out_op = _W['out'].assign(initial_weight_matrix_2_out, use_locking=False)
    sess.run(assign_weight2out_op)
    # # tf.nn.dropout(layer_2, 0.5)
    return final_indices
    # return tf.matmul(layer_2, _W['out']) + _biases['out']

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
            _,loss_val=sess.run([optimizer,loss], feed_dict={x: batch_x, y: batch_y})
            #if cost(W)<=threshold. Threshold selected as 0.02
            if loss_val<=0.20:
                # print(loss_val)
                print('Start')
                # bias_test0 = tf.Print(biases['fc1'],[biases['fc1']], summarize = 900)
                weight_test0 = tf.Print(W['fc1'],[W['fc1']], summarize = 900)                

                # print(sess.run(bias_test0))
                print(sess.run(weight_test0))

                pred1=model_prune(batch_x, W, biases)
                print('pred')
                print(sess.run([pred1]))

                # bias_test = tf.Print(biases['fc1'],[biases['fc1']], summarize = 900)
                # weight_test = tf.Print(W['fc1'],[W['fc1']], summarize = 900)

                # print(sess.run(weight_test))

                w_fc1, w_fc2, w_out = sess.run([W['fc1'],W['fc2'],W['out']])
                sparsity = np.count_nonzero(w_fc1)
                sparsity += np.count_nonzero(w_fc2)
                sparsity += np.count_nonzero(w_out)
                num_parameter = np.size(w_fc1)
                num_parameter += np.size(w_fc2)
                num_parameter += np.size(w_out)
                total_sparsity = float(sparsity)/float(num_parameter)
                print ("Total Sparsity= ", sparsity, "/", num_parameter, \
                " = ", total_sparsity*100, "%")
                print ("Compression Rate = ", float(num_parameter)/float(sparsity))


                print('end')

                # print(sess.run(W['fc1']))


        

        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
