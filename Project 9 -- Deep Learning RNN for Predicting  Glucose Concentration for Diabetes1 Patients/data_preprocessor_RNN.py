import csv
import re
import os
import random
from datetime import datetime
from datetime import timedelta
from glob import glob
#import dateutil.parser # For parsing the date/time format used in the raw data
PREDICTION_HORIZON = 20
inputFolder = 'tblADataRTCGM_Unblinded_ControlGroup_1'
patients = glob(inputFolder + '/*')
outputFolder = inputFolder + '_output_RNN_' + str(PREDICTION_HORIZON)
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)


#
# Goal is to create data points consisting of a blood sugar measurement at a
# prediction time horizon of 20 mins, and measurements at the following times:
#   - Current time
#   - Current time - 10 mins
#   - Current time - 20 mins
#   - Current time - 30 mins
#   - Current time - 40 mins
#   - Current time - 50 mins
#   - Current time - 60 mins
#
# When the specified measurement time does not coincide with a measurement,
# the blood sugar level at the specified time will be determined by linear
# interpolation of the two nearest measurement points.
#
# Measurement points will only be used if they meet the following criteria:
#   - Values determined from linear interpolation must be determined from
#       measurements separated by no more than 11 mins
#   - All values must be taken from between the hours of 11pm and 7am. In other
#       words, the first prediction time considered will be ~1am.
#

# Performs a linear interpolation of the blood sugar measurements between two
# input data points.
def interpolate(prev, cur, desiredTime) :
    totalTime = (cur[0] - prev[0]).total_seconds()
    prevToDesiredTime = (desiredTime - prev[0]).total_seconds()
    # Perform linear interpolation:
    sugar = prevPoint[1] + (((curPoint[1] - prevPoint[1]) / totalTime) * prevToDesiredTime)
    return sugar

# Read input file
for patientFolder in patients :
    patientID = re.split("/", patientFolder)[1]
    outputTrainFile = outputFolder + '/' + patientID + '_train.csv'
    outputTestFile = outputFolder + '/' + patientID + '_test.csv'
    open(outputTrainFile, 'w').close()
    open(outputTestFile, 'w').close()
    days = glob(patientFolder + '/*')
    for dayFile in days :
        # Create new empty lists
        data = []
        processedData_Train = []
        processedData_Test = []
        with open(dayFile, newline='') as f:
            reader = csv.reader(f)
            # Populate the list with a full day worth of data
            # (from 12 noon until 12 noon to ensure that the complete night is included)
            for row in reader :
                #curTime = dateutil.parser.parse(row[2]) # Parse date
                #curTime = datetime.strptime(row[2], '%Y-%m-%d %H:%M:%S.%f') # Example date: "2001-03-03 12:01:00.000000"
                #bloodSugar = float(row[3]) # Parse blood sugar
                curTime = datetime.strptime(row[0], '%m\%d\%Y %H:%M')
                bloodSugar = float(row[1])
                newTuple = (curTime, bloodSugar)
                data.append(newTuple)
        # End with file

        # Make sure data is sorted by time:
        data.sort(key=lambda tup: tup[0])

        # For each data point:
        for index, predictionPoint in enumerate(data) :
            valid = 1
            # Only consider points between 1AM and 7AM:
            if predictionPoint[0].hour >= 1 and predictionPoint[0].hour < 7 :
                # Determine all desired measurement times:
                time1 = predictionPoint[0] - timedelta(minutes = PREDICTION_HORIZON) # 30 mins ago
                tenMinutes = timedelta(minutes = 10)
                time2 = time1 - tenMinutes # 40 mins ago
                time3 = time2 - tenMinutes # 50 mins ago
                time4 = time3 - tenMinutes # 60 mins ago
                time5 = time4 - tenMinutes # 70 mins ago
                time6 = time5 - tenMinutes # 80 mins ago
                time7 = time6 - tenMinutes # 90 mins ago

                # Initialize all blood sugar measurements
                bloodSugar1 = -1.0
                bloodSugar2 = -1.0
                bloodSugar3 = -1.0
                bloodSugar4 = -1.0
                bloodSugar5 = -1.0
                bloodSugar6 = -1.0
                bloodSugar7 = -1.0

                curPoint = predictionPoint
                # Iterate from current index backwards
                for i in range(index - 1, -1, -1) :
                    prevPoint = data[i]
                    # If prevPoint and curPoint aren't too far apart
                    if (prevPoint[0] + timedelta(minutes = 12)) > curPoint[0] :
                        if prevPoint[0] <= time1 and curPoint[0] > time1 : # Straddling time1
                            bloodSugar1 = interpolate(prevPoint, curPoint, time1)
                        elif prevPoint[0] <= time2 and curPoint[0] > time2 : # Straddling time2
                            bloodSugar2 = interpolate(prevPoint, curPoint, time2)
                        elif prevPoint[0] <= time3 and curPoint[0] > time3 : # Straddling time3
                            bloodSugar3 = interpolate(prevPoint, curPoint, time3)
                        elif prevPoint[0] <= time4 and curPoint[0] > time4 : # Straddling time4
                            bloodSugar4 = interpolate(prevPoint, curPoint, time4)
                        elif prevPoint[0] <= time5 and curPoint[0] > time5 : # Straddling time5
                            bloodSugar5 = interpolate(prevPoint, curPoint, time5)
                        elif prevPoint[0] <= time6 and curPoint[0] > time6 : # Straddling time6
                            bloodSugar6 = interpolate(prevPoint, curPoint, time6)
                        elif prevPoint[0] <= time7 and curPoint[0] > time7 : # Straddling time7
                            bloodSugar7 = interpolate(prevPoint, curPoint, time7)
                    curPoint = prevPoint # Update curPoint for next iteration

                # If all bloodSugar measuremets were determined:
                if  (bloodSugar1 > 0) and (bloodSugar2 > 0) and (bloodSugar3 > 0) and (bloodSugar4 > 0) and (bloodSugar5 > 0) and (bloodSugar6 > 0) and (bloodSugar7 > 0) :
                    newPoint = (bloodSugar7, bloodSugar6, bloodSugar5, bloodSugar4, bloodSugar3, bloodSugar2, bloodSugar1, predictionPoint[1])
                    randNum = random.random() #Random float between 0 and 1
                    if randNum > 0.25 :
                        processedData_Train.append(newPoint) # Added to training set with probability of 75%
                    else :
                        processedData_Test.append(newPoint)
            # End if predictionPoint is between 1AM and 7AM
        # End for each input data predictionPoint

        # Write results to output files:
        with open(outputTrainFile, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(processedData_Train)

        with open(outputTestFile, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(processedData_Test)
