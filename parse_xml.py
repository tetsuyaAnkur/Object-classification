import xml.etree.ElementTree
import os

def parsing(a,b):

	l = {}

	for file in os.listdir(a):
		key = b + "/" + file[0:len(file)-4] + ".jpg"
		root = xml.etree.ElementTree.parse(a+"/"+file).getroot()
		m = []
		for x in root.findall('object'):
			category = x.find('name').text
			m.append(category)
			
		l[key] = m
	return(l)

#print(l)
#print(len(l.keys()))
