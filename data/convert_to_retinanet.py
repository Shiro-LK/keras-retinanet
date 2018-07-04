# -*- coding: utf-8 -*-

import csv
import cv2
import numpy as np
from optparse import OptionParser

def ConvertToRetinaNetFormatCSV(filename):
    '''
        Convert dataset in txt format to csv format for retinanet model.
        converter works on windows. For linux user, remove the delimiter parameters.
    '''
    found_bg = False
    writer_train = csv.writer(open(filename.replace('.txt', '_train.csv'), "w", newline=''), delimiter=",")
    writer_test = csv.writer(open(filename.replace('.txt', '_test.csv'), "w", newline=''), delimiter=",")
    writer_class = csv.writer(open(filename.replace('.txt', '_classes.csv'), "w", newline=''), delimiter=",")
    class_mapping = {}
    cpt_train = 0
    cpt_test = 0
    class_train_cpt = {}
    class_test_cpt = {}
    class_mapping = {}
    prev = None
    with open(filename,'r') as f:
        for line in f:
            line_split = line.strip().split(',')
            (filename,width, height, x1,y1,x2,y2,class_name, imageset) = line_split
            if imageset == 'training':
                
                if filename != prev:
                        cpt_train += 1
                        prev = filename
                        
                writer_train.writerow([filename,x1,y1,x2,y2,class_name])
                
                if class_name in class_train_cpt:
                        class_train_cpt[class_name] += 1
                else:
                        class_train_cpt[class_name] = 1
            else:
                
                if filename != prev:
                        cpt_test += 1
                        prev = filename
                        
                writer_test.writerow([filename,x1,y1,x2,y2,class_name])
                
                if class_name in class_test_cpt:
                        class_test_cpt[class_name] += 1
                else:
                        class_test_cpt[class_name] = 1
                        
            if class_name not in class_mapping:
                if class_name == 'bg' and found_bg == False:
                    print('Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
                    found_bg = True
                class_mapping[class_name] = len(class_mapping)  
    for class_, idx_ in class_mapping.items():
        writer_class.writerow([class_, idx_])
    print('number image training :', cpt_train)
    print('number image test :', cpt_test)
    print('number of label in train :', class_train_cpt)
    print('number of label in test :', class_test_cpt)    
        
####### MAIN ########
    
parser = OptionParser()
parser.add_option('-f', "--filename", dest="filename", help='.txt filename to convert in .csv format (retinanet)', default='raccoon.txt')        
(options, args) = parser.parse_args()
ConvertToRetinaNetFormatCSV(options.filename)
