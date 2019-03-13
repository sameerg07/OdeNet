import os

allImageFolders = os.listdir("./originalGrimace/")

allImageFolders.sort()

i=0
for folder in allImageFolders:
	listOfIms = os.listdir("./originalGrimace/"+folder+"/")
	listOfIms.sort()
	for file in listOfIms:
		fileCopy = file
		os.rename("./originalGrimace/"+folder+"/"+file,"./originalGrimace/"+folder+"/"+str(i)+"_"+fileCopy)
	i+=1