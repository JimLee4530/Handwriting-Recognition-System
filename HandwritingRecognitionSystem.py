from numpy import *
from os import listdir
import kNN
import operator

def img2verctor(filename):
	returnVect = zeros((1,1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0,32*i+j] = int(lineStr[j])
	return returnVect

def handwritingClassTest():
	hwLabels = []
	trainingFileList = listdir('/home/jimlee/Documents/Git/kNN/HandwritingRecognitionSystem/trainingDigits')
	m = len(trainingFileList)
	trainingMat = zeros((m,1024))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		hwLabels.append(classNumStr)
		trainingMat[i,:] = img2verctor('/home/jimlee/Documents/Git/kNN/HandwritingRecognitionSystem/trainingDigits/%s' % fileNameStr)
	testFileList = listdir('/home/jimlee/Documents/Git/kNN/HandwritingRecognitionSystem/testDigits')
	errorCount = 0.0
	m_test = len(testFileList)
	for i in range(m_test):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2verctor('/home/jimlee/Documents/Git/kNN/HandwritingRecognitionSystem/testDigits/%s' % fileNameStr)
		classifierResult = kNN.classify0(vectorUnderTest,trainingMat,hwLabels,3)
		print "the classifier came back with: %d, the real answer is: %d" %(classifierResult,classNumStr)
		if (classifierResult != classNumStr):
			errorCount += 1.0
	print "\nthe total number of errors is: %d" % errorCount
	print "\nthe total error rate is: %f" % (errorCount/float(m_test))



if __name__ == '__main__':
	handwritingClassTest()