def preprocess():
	import os
	import cv2
	import numpy as np
	import time
	import random
	data={}
	for filename in os.listdir(os.getcwd()+'/JPEGImages'):
		data[filename]=cv2.imread(os.getcwd()+'/JPEGImages/'+filename)
	#print(len(data.keys()))
	max_h,max_w=0,0
	for key in data.keys():
		if(data[key].shape[0]>max_h):
			max_h=data[key].shape[0]
		if(data[key].shape[1]>max_w):
			max_w=data[key].shape[1]
	#print(max_h,max_w)
	datanew={}
	for key in data.keys():
		#print(data[key].shape)
		#print(data[key].dtype)	
		datanew[key]=np.zeros((max_h,max_w,3),dtype=np.uint8)
		#print(datanew[key].dtype)	
		datanew[key][:data[key].shape[0],:data[key].shape[1],:]=data[key]
	del data
	alldata=random.sample(datanew.keys(),3000)
	import parse_xml as pxml
	l=pxml.parsing(os.getcwd()+'/Annotations',os.getcwd()+'/JPEGImages')
	classes={}
	for x in datanew.keys():
		for x in l[os.getcwd()+'/JPEGImages/'+x]:
			if x not in classes.keys():
				classes[x]=1
	setting={}
	i=0
	order=classes.keys()
	order.sort()
	file1=open('classes.txt','w')
	for x in order:
		setting[x]=i
		i=i+1
		file1.write(x)
	file1.close()
	labels={}
	for x in datanew.keys():
		temp=np.zeros(20)
		for y in l[os.getcwd()+'/JPEGImages/'+x]:
			temp[setting[y]]=1
		#print(l[os.getcwd()+'/JPEGImages/'+x])
		labels[x]=temp
		#print(temp)
	
	train_data=[]
	train_labels=[]
	for x in alldata[:int(0.9*3000)]:
		train_data.append(datanew[x])
		train_labels.append(labels[x])
	test_data=[]
	test_labels=[]
	for x in alldata[int(0.9*3000):]:
		test_data.append(datanew[x])
		test_labels.append(labels[x])
	del datanew
	del labels
	print(setting)
	return (np.asarray(train_data),np.asarray(train_labels),np.asarray(test_data),np.asarray(test_labels))

