#
# CISC 452
# Prediction of Blood Glusose Levels based on RTCGM Data
#
# November 10, 2016
#
# This script implements a multi-layer feed-forward neural network for glucose
# prediciton
#
# The network has 7 input nodes and 1 output node. If the current time is 'T',
# then the inputs and output represent the blood glucose measurements at the
# following times:
#   Inputs:     - T
#               - (T - 10 mins)
#               - (T - 20 mins)
#               - (T - 30 mins)
#               - (T - 40 mins)
#               - (T - 50 mins)
#               - (T - 60 mins)
#
#   Output:     - (T + 20 mins)
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import matplotlib.pyplot as plt

NUM_EPOCHS = 1500 # Number of training epochs

# readData reads data from the specified pre-processed input data file.
# The function returns an array of input data points and an array of the
# corresponding desired outputs.
def readData(filePath) :
    x_data = []
    y_data = []
    with open(filePath, 'r') as f:
        for line in f:
            values = line.split(',')
            time1 = float(values[0])
            time2 = float(values[1])
            time3 = float(values[2])
            time4 = float(values[3])
            time5 = float(values[4])
            time6 = float(values[5])
            time7 = float(values[6])
            time8 = float(values[7])
            newPointx = [time1, time2, time3, time4, time5, time6, time7] # Input
            newPointy = [time8] # Desired Output
            x_data.append(newPointx)
            y_data.append(newPointy)
    data = [x_data, y_data]
    return data;

# evaluateNetwork runs the trained network on the the provided network and
# reports the following evaluation metrics:
#   - mean squared prediction error
#   - percentage of lows that were correctly identified
#   - percentage of highs that were corretly identified
#   - number of falsely reported lows
#   - number of falsely reported highs
#
# These metrics are defined as follows:
#   - MSE:
#       -> Average of (y_desired - y_actual)^2 for each test point
#   - Low prediction accuracy:
#       -> 100 * (Number of correct lows) / (Number of lows)
#       -> Lows are any blood glucose level less than 70 mg/dL
#   - High prediction accuracy:
#       -> 100 * (Number of correct highs) / (Number of highs)
#       -> Highs are any blood glucose level greater than 200
#   - Number of false lows:
#       -> Number of false lows where (y_desired - y_actual) > 6
#       -> Note: false alarms are not counted if the prediction error is small
#   - Number of false highs:
#       -> Number of false highs where (y_actual - y_desired) > 6
#       -> Note: false alarms are not counted if the prediciton error is small
def evaluateNetwork(session, inData, outData, prediction) :
    # Compute mse:
    mse = session.run(tf.reduce_mean(tf.square(prediction - y_desired)), feed_dict={x: inData, y_desired: outData})
    numTestPoints = len(inData)
    numPredictedLows = 0
    numLows = 0
    numFalseLows = 0
    numPredictedHighs = 0
    numHighs = 0
    numFalseHighs = 0
    for i, inputPoint in enumerate(inData) :
        # Apply network on current point:
        predicted = session.run(prediction, feed_dict={x: [inputPoint]})
        desired = outData[i][0]


        # Update numLows, numHighs:
        if(desired < 70) :
            numLows += 1
        elif(desired > 200) :
            numHighs += 1

        # Update prediction counts:
        if(predicted < 70) : # If predicted low
            if(desired < 70) : # If low prediction was correct
                numPredictedLows += 1
            elif((desired - predicted) > 8) : # If low prediction was incorrect and error was 'large'
                numFalseLows += 1
        elif(predicted > 200) : # If predicted high
            if(desired > 200) : # If high prediction was correct
                numPredictedHighs += 1
            elif((predicted - desired) > 8) : # If high prediction was incorrect and error was 'large'
                numFalseHighs += 1

    # Print results:
    print('Number of test points: ', numTestPoints)
    print('Number of lows: ', numLows)
    print('Number of highs: ', numHighs)
    print("Number of 'normal' points: ", numTestPoints - numLows - numHighs)
    print('') # New line
    print('MSE: ', mse)
    print('')
    print('Low prediction accuracy: ', 100 * numPredictedLows / numLows, '%')
    print('Number of false lows: ', numFalseLows)
    print('')
    print('High prediction accuracy: ', 100 * numPredictedHighs / numHighs, '%')
    print('Number of false highs: ', numFalseHighs)
# End evaluateNetwork(...)

x = tf.placeholder(tf.float32, [None, 7], name='x') # Input placeholder
y_desired = tf.placeholder(tf.float32, [None, 1], name='y_desired') # Desired output placeholder

# feedForwardNN describes the model of the feed forward neural network being
# used. The selected architecture consists of two hidden layers containing 15
# nodes each. All nodes employ a linear activation function.
def feedForwardNN(x) :
    # Weights from inputs to first hidden layer (15 nodes):
    Wh1 = tf.Variable(tf.random_uniform([7, 15], minval = -1, maxval = 1, dtype = tf.float32))
    # Bias for first hidden layer:
    bh1 = tf.Variable(tf.zeros([1, 15]))

    # Weights from first hidden layer to second (15 nodes):
    Wh2 = tf.Variable(tf.random_uniform([15, 15], minval = -1, maxval = 1, dtype = tf.float32)) # The weights from each of the 784 inputs to the 10 output nodes
    # Bias for second hidden layer:
    bh2 = tf.Variable(tf.zeros([1, 15])) # One bias input for each of the 10 output nodes

    # Weights from second hidden layer to output layer (1 node):
    Wo = tf.Variable(tf.random_uniform([15, 1], minval = -1, maxval = 1, dtype = tf.float32))
    # Bias to output node:
    bo = tf.Variable(tf.zeros([1, 1]))

    # Nodes have no output function (they simply output their activation):
    h1 = tf.add(tf.matmul(x, Wh1), bh1) # Hidden layer 1 output
    h2 = tf.add(tf.matmul(h1, Wh2), bh2) # Hidden layer 2 output
    output = tf.add(tf.matmul(h2, Wo), bo) # Network output

    return output

def trainFFNN(x):
    # Import the training data and test data:
    # 151, 149, 174
    trainData_in, trainData_out = readData('tblADataRTCGM_Unblinded_ControlGroup_1_output_RNN_20/174_train.csv')
    testData_in, testData_out = readData('tblADataRTCGM_Unblinded_ControlGroup_1_output_RNN_20/174_test.csv')

    prediction = feedForwardNN(x)

    # Error function to be minimized is the mean square error:
    loss = tf.reduce_mean(tf.square(prediction - y_desired))

    # Define training algorithm (Adam Optimizer):
    # Note: AdamOptimizer produced better results than the GradientDescentOptimizer
    #train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    train_step = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

    # Train:
    errors = []
    sess = tf.InteractiveSession()
    tf.initialize_all_variables().run()
    for i in range(NUM_EPOCHS): # 1000 training epochs
        ### Batch training was tested, but per-epoch produced better results:
        # Train with one batch at a time:
        #for start, end in zip(range(0, len(trainData_in), BATCH_SIZE), range(BATCH_SIZE, len(trainData_in), BATCH_SIZE)):
        #    sess.run(train_step, feed_dict={x: trainData_in[start:end], y_desired: trainData_out[start:end]})

        # Per-Epoch training:
        sess.run(train_step, feed_dict={x: trainData_in, y_desired: trainData_out})
        # Print MSE on test data after every 10 epochs
        # i % 10 == 0 :
        #    mse = sess.run(tf.reduce_mean(tf.square(prediction - y_desired)), feed_dict={x: testData_in, y_desired: testData_out})
        #    errors.append(mse)
        #    print(mse)

    # Output the desired and actual outputs for each test data point
    #for i, inputPoint in enumerate(testData_in) :
    #    output = sess.run(y, feed_dict={x: [inputPoint]})
    #    print('desired: ', testData_out[i], ', actual: ', output)

    # Test:
    print('Patient 174 data:')
    evaluateNetwork(sess, testData_in, testData_out, prediction)
    print('Patient 149 data:')
    testData_in, testData_out = readData('tblADataRTCGM_Unblinded_ControlGroup_1_output_RNN_20/149_test.csv')
    evaluateNetwork(sess, testData_in, testData_out, prediction)
    print('Patient 151 data:')
    testData_in, testData_out = readData('tblADataRTCGM_Unblinded_ControlGroup_1_output_RNN_20/151_test.csv')
    evaluateNetwork(sess, testData_in, testData_out, prediction)
    # Uncomment this to evaluate the current network on a different patient:
    #testData_in, testData_out = readData('tblADataRTCGM_Blind_Baseline_Split_output/78_test.csv')
    #evaluateNetwork(sess, testData_in, testData_out, prediciton)

    # Plot the MSE throughout training
    #plt.plot(errors)
    #plt.xlabel('#epochs')
    #plt.ylabel('MSE')
    #plt.show()
# End trainFFNN(x)

trainFFNN(x)
