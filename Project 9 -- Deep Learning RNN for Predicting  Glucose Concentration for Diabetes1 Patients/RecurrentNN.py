import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
import matplotlib.pyplot as plt

NUM_EPOCHS = 500
batch_size = 128
chunk_size = 1
n_chunks = 7
rnn_size = 25

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
    mse = session.run(tf.reduce_mean(tf.square(prediction - y)), feed_dict={x: inData, y: outData})
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

x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float')

# recurrent_neural_network() defines the RNN model. The network architecture
# used consists of a single LSTM cell followed by an output layer.
def recurrent_neural_network(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size, 1])),
             'biases':tf.Variable(tf.random_normal([1]))}

    # Reshape x to the format desired by the LSTM:
    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(0, n_chunks, x)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size, state_is_tuple=True, activation=tf.nn.relu)
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

    return output

# The RNN is trained by feeding in sequences of glucose measurements seprarated
# by 10 minute intervals, and the desired output at a 20 minute prediction horizon.
def train_neural_network(x):
    trainData_in, trainData_out = readData('tblADataRTCGM_Unblinded_ControlGroup_1_output_RNN_20/174_train.csv')
    testData_in, testData_out = readData('tblADataRTCGM_Unblinded_ControlGroup_1_output_RNN_20/174_test.csv')
    trainData_in = np.reshape(trainData_in, [-1,n_chunks,chunk_size])
    testData_in = np.reshape(testData_in, [-1,n_chunks,chunk_size])
    prediction = recurrent_neural_network(x)

    # Use MSE as cost function to be minimized
    cost = tf.reduce_mean(tf.square(prediction - y))

    # AdamOptimizer produced better results than simple GradientDescentOptimizer
    optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

    errors = []
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        # Per-epoch training:
        for i in range(NUM_EPOCHS):
            sess.run(optimizer, feed_dict={x: trainData_in, y: trainData_out})
            #if i % 10 == 0 :
            #    mse = sess.run(tf.reduce_mean(tf.square(prediction - y)), feed_dict={x: testData_in, y: testData_out})
            #    errors.append(mse)
            #    print(mse)

        print('Patient 174 data:')
        evaluateNetwork(sess, testData_in, testData_out, prediction)
        print('Patient 149 data:')
        testData_in, testData_out = readData('tblADataRTCGM_Unblinded_ControlGroup_1_output_RNN_20/149_test.csv')
        testData_in = np.reshape(testData_in, [-1,n_chunks,chunk_size])
        evaluateNetwork(sess, testData_in, testData_out, prediction)
        print('Patient 151 data:')
        testData_in, testData_out = readData('tblADataRTCGM_Unblinded_ControlGroup_1_output_RNN_20/151_test.csv')
        testData_in = np.reshape(testData_in, [-1,n_chunks,chunk_size])
        evaluateNetwork(sess, testData_in, testData_out, prediction)
        # Uncomment this to evaluate the current network on a different patient:
        #testData_in, testData_out = readData('tblADataRTCGM_Blind_Baseline_Split_output/78_test.csv')
        #evaluateNetwork(sess, testData_in, testData_out, prediciton)

        # Plot the MSE throughout training
        #plt.plot(errors)
        #plt.xlabel('#epochs')
        #plt.ylabel('MSE')
        #plt.show()
#End train_neural_network(x)

train_neural_network(x)
