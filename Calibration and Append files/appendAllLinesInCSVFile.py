# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 23:05:04 2019
Appens a value in an ID file
@author: mfm160330
"""
import csv 

idFile = 'C:/AdasData/FaceAndEyes/FE2018-12-3/id -RemoveMisclassified.csv' #'id.csv'
writePath = 'C:/AdasData/FaceAndEyes/FE2018-12-3/idNewFormat.csv'
IDs = []
with open(idFile, "r") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        if not ''.join(row).strip():
            continue # ignore the blank lines
        #IDs.append(row[0])
        #labels.append(row[1])
        row = ['FE2018-12-3'] + row 
        IDs.append(row)
        
for row2 in IDs:
    with open(writePath, 'a+') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)            
        filewriter.writerow(row2)


