import os
import pickle
import sys
import random
import cv2
import numpy as np
import tensorflow as tf
import pyimgsaliency as psal
# from shutil import copy
import scipy.misc
import numpy as np
import sys, getopt
from glob import glob

datasetFolderName = "originalGrimace" 
# testFolder = "test_data"
# trainFolder = "train_data"
# dumpFolder = "dump"

def load_data():
	allImageFolders = os.listdir("./"+datasetFolderName+"/")
	numOfFilesInDataset = len(allImageFolders)
	# try:
	# 	for i in range(0,numOfFilesInDataset):
	# 		os.makedirs(os.getcwd()+"/"+testFolder+"/"+str(i)+"/")
	# 		os.makedirs(os.getcwd()+"/"+trainFolder+"/"+str(i)+"/")
	# except:
	# 	pass

	listOfImgNames = []
	listOfImgPaths = []
	# xTrain,xTest,yTrain,yTest = np.empty((306,64,64),dtype="uint8"),np.empty((54,64,64),dtype="uint8"),np.empty((306,),dtype="uint8"),np.empty((54,),dtype="uint8")
	trainX,trainY = [],[]
	testX,testY = [],[]
	finalRes = tuple()

	for folder in allImageFolders:
		setOfFiles = os.listdir(os.path.join(os.getcwd()+"/"+datasetFolderName+"/"+folder+"/"))
		# random.shuffle(setOfFiles)
		listOfImgNames.append(setOfFiles)
		temp = [os.getcwd()+"/"+datasetFolderName+"/"+folder+"/"+str(ele) for ele in setOfFiles]
		listOfImgPaths.append(temp)

	sess = tf.Session()	
	# print(listOfImgPaths)
	for setOfPaths in listOfImgPaths:
		# for image in setOfPaths:
		for i in  range(0,len(setOfPaths)-3):
			# below two lines are Original
			bgrImage = cv2.imread(setOfPaths[i])
			# bgrImage = psal.get_saliency_mbd(setOfPaths[i]).astype('uint8')
			iiImage = cv2.cvtColor(bgrImage,cv2.COLOR_BGR2GRAY)
			# im = im.tolist()
			# end of original lines
			
			# start of lines added for PCA II			
			# bgrImage = get_image(setOfPaths[i])
			# iiImages,alpha = pca_ii(bgrImage)
			# iiImage ,angle = sess.run([iiImages, alpha])
			# end of lines added for PCA II			
			
			trainX.append(iiImage)
			res = setOfPaths[i].split('/')
			res = res[len(res)-1]
			res = int(res.split('_')[0])
			# print("res:",res)
			trainY.append(res)
		for j in range(17,20):
			# below two lines are Original
			bgrImage = cv2.imread(setOfPaths[j])
			# bgrImage = psal.get_saliency_mbd(setOfPaths[j]).astype('uint8')
			iiImage = cv2.cvtColor(bgrImage,cv2.COLOR_BGR2GRAY)
			# end of original lines

			# start of lines added for PCA II			
			# bgrImage = get_image(setOfPaths[i])
			# iiImages,alpha = pca_ii(bgrImage)
			# iiImage ,angle = sess.run([iiImages, alpha])
			# end of lines added for PCA II			

			testX.append(iiImage)
			res = setOfPaths[j].split('/')
			res = res[len(res)-1]
			res = int(res.split('_')[0])
			testY.append(res)
	trainX = np.asarray(trainX,dtype="uint8")
	trainY = np.asarray(trainY,dtype="uint8")
	testX = np.asarray(testX,dtype="uint8")
	testY = np.asarray(testY,dtype="uint8")
	# np.random.shuffle(trainX)
	# np.random.shuffle(trainY)
	# np.random.shuffle(testX)
	# np.random.shuffle(testY)
	# print(trainX)
	# print(trainY)
	finalRes = ((trainX,trainY),(testX,testY))
	# print(finalRes)
	# print(testX)
	# print(testY)

	# print("Len of trainX: ",len(trainX))
	# print("Len of trainY: ",len(trainY))
	# print("Len of testX: ",len(testX))
	# print("Len of testY: ",len(testY))
	# print("Shape of trainX: ",trainX.shape)
	# print("Shape of trainY: ",trainY.shape)
	# print("Shape of testX: ",testX.shape)
	# print("Shape of testY: ",testY.shape)
	return finalRes

if __name__ == '__main__':
	# result = Result(load_data())
	tempRes = load_data()
	result = {"res":tempRes}
	with open('grimaceData.pickle', 'wb') as fileHandle:
		pickle.dump(result, fileHandle, protocol=pickle.HIGHEST_PROTOCOL)