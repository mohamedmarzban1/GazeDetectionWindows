# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:27:16 2019
Restore shelved data test
@author: mfm160330
"""
import shelve

ShelveFilename = 'variables/run1.out'
my_shelf = shelve.open(ShelveFilename)
for key in my_shelf:
    globals()[key]=my_shelf[key]
my_shelf.close()

