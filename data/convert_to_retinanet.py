# -*- coding: utf-8 -*-

import csv
import cv2
import numpy as np
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
    with open(filename,'r') as f:
        for line in f:
            line_split = line.strip().split(',')
            (filename,width, height, x1,y1,x2,y2,class_name, imageset) = line_split
            if imageset == 'training':
                writer_train.writerow([filename,x1,y1,x2,y2,class_name])
            else:
                writer_test.writerow([filename,x1,y1,x2,y2,class_name])
            if class_name not in class_mapping:
                if class_name == 'bg' and found_bg == False:
                    print('Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
                    found_bg = True
                class_mapping[class_name] = len(class_mapping)  
    for class_, idx_ in class_mapping.items():
        writer_class.writerow([class_, idx_])
        
def Convertjpgtopickle(filename, directory_input, directory,channels=3, output_format='.npz'):
     '''
        convert dataset in jpg format to pkl format for retinanet (multi channels)
     '''
     found_bg = False
     writer_train = csv.writer(open(filename.replace('.txt', '_train.csv'), "w", newline=''), delimiter=",")
     writer_test = csv.writer(open(filename.replace('.txt', '_test.csv'), "w", newline=''), delimiter=",")
     writer_class = csv.writer(open(filename.replace('.txt', '_classes.csv'), "w", newline=''), delimiter=",")
     class_mapping = {}
     with open(filename,'r') as f:
        for line in f:
            line_split = line.strip().split(',')
            (name,width, height, x1,y1,x2,y2,class_name, imageset) = line_split
            img = cv2.imread(directory_input+name)
            if img is None:
                print("can't load name")
            np.savez_compressed(directory+name.replace('.jpg', output_format), img)

            name = name.replace('.jpg', output_format)
            if imageset == 'training':
                writer_train.writerow([name,width, height,x1,y1,x2,y2,class_name])
            else:
                writer_test.writerow([name,width, height, x1,y1,x2,y2,class_name])
            if class_name not in class_mapping:
                if class_name == 'bg' and found_bg == False:
                    print('Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
                    found_bg = True
                class_mapping[class_name] = len(class_mapping)  
     for class_, idx_ in class_mapping.items():
        writer_class.writerow([class_, idx_])
            
def ConvertRaccoontopickle(filename, directory_input, directory,channels=3, output_format='.npz'):
     '''
        convert raccoon dataset in jpg format to pkl format for retinanet (multi channels = 4)
     '''
     found_bg = False
     writer_train = csv.writer(open(filename.replace('.txt', '_train4.csv'), "w", newline=''), delimiter=",")
     writer_test = csv.writer(open(filename.replace('.txt', '_test4.csv'), "w", newline=''), delimiter=",")
     writer_class = csv.writer(open(filename.replace('.txt', '_classes4.csv'), "w", newline=''), delimiter=",")
     class_mapping = {}
     with open(filename,'r') as f:
        for line in f:
            line_split = line.strip().split(',')
            (name,width, height, x1,y1,x2,y2,class_name, imageset) = line_split
            img = cv2.imread(directory_input+name)
            img2 = np.expand_dims(np.mean(img, axis=-1), axis=-1)
            #print(img.shape, img2.shape)
            img = np.concatenate([img, img2], axis=-1)
            #print(img.shape)
            if img is None:
                print("can't load name")
            np.savez_compressed(directory+name.replace('.jpg', output_format), img)

            name = name.replace('.jpg', output_format)
            if imageset == 'training':
                writer_train.writerow([name,width, height,x1,y1,x2,y2,class_name])
            else:
                writer_test.writerow([name,width, height, x1,y1,x2,y2,class_name])
            if class_name not in class_mapping:
                if class_name == 'bg' and found_bg == False:
                    print('Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
                    found_bg = True
                class_mapping[class_name] = len(class_mapping)  
     for class_, idx_ in class_mapping.items():
        writer_class.writerow([class_, idx_])    
ConvertToRetinaNetFormatCSV('coco2017+VOC.txt')
#Convertjpgtopickle('raccoon.txt', 'raccoon/', directory='raccoon_pkl/')
        
#ConvertRaccoontopickle('raccoon.txt', 'raccoon/', directory='raccoon_4/')
