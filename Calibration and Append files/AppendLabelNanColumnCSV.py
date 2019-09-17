# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 22:27:21 2019
Append label:nan column in csv
@author: mfm160330
"""

import csv

AppendItem = "nan"
fileInputPath = 'G:/ContGazeImages/FaceAndEyes/CFE2019-5-30/CFE2019-5-30FinalFormatIDFile.csv'
fileOutputPath = 'G:/ContGazeImages/FaceAndEyes/CFE2019-5-30/CFE2019-5-30FinalFormatIDFile2.csv'


with open(fileInputPath,'r') as csvinput, open(fileOutputPath, 'w') as csvoutput:
        #filewriter = csv.writer(csvoutput, delimiter='\t', quotechar='', quoting=csv.QUOTE_NONE, escapechar='\n')
        filereader = csv.reader(csvinput)

        #all = []
        row = next(filereader)
        row[0] = row[0]+'\tlabels\n'
        csvoutput.write(row[0])

        #all.append(row[0])

        for row in filereader:
            if not ''.join(row).strip():
                continue # ignore the blank lines
            row[0] = row[0]+'\t'+AppendItem+'\n'
            #all.append(row)
            csvoutput.write(row[0])
            csvoutput.write('\n')

        #filewriter.writerows(all)