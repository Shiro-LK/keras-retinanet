# -*- coding: utf-8 -*-

import csv
import cv2
import numpy as np
from optparse import OptionParser
def ConvertToRetinaNetFormatMultiCSV(filename, directory='', new='features', channels=3, output_format='.npz', convert_to_npz=True):
    '''
        Convert dataset in txt format to csv format for retinanet model.
        Concatenate images in .npz format.
    '''
    found_bg = False
    writer_train = csv.writer(open(filename.replace('.txt', '_train.csv'), "w", newline=''), delimiter=",")
    writer_test = csv.writer(open(filename.replace('.txt', '_test.csv'), "w", newline=''), delimiter=",")
    writer_class = csv.writer(open(filename.replace('.txt', '_classes.csv'), "w", newline=''), delimiter=",")
    cpt_train = 0
    cpt_test = 0
    class_train_cpt = {}
    class_test_cpt = {}
    class_mapping = {}
    prev = None
    count = 0
    with open(filename,'r') as f:
        for line in f:
            line_split = line.strip().split(',')
            if channels == 1:
                (filename1,width, height, x1,y1,x2,y2,class_name, imageset) = line_split
                filename = [filename1]    
            elif channels == 2:
                (filename1, filename2, width, height, x1,y1,x2,y2,class_name, imageset) = line_split
                filename = [filename1, filename2]
            elif channels == 3:
                (filename1, filename2,filename3,width, height, x1,y1,x2,y2,class_name, imageset) = line_split
                filename = [filename1,filename2,filename3]
            elif channels == 4:
                (filename1, filename2,filename3,filename4,width, height, x1,y1,x2,y2,class_name, imageset) = line_split
                filename = [filename1,filename2,filename3,filename4]
            elif channels == 5:
                (filename1, filename2,filename3,filename4,filename5,width, height, x1,y1,x2,y2,class_name, imageset) = line_split
                filename = [filename1,filename2,filename3,filename4,filename5]
             
                
            idx = filename1.rfind('_')
            newname = filename1[:idx+1]+new+output_format    
            if imageset == 'training':
                    writer_train.writerow([newname,width, height,x1,y1,x2,y2,class_name])
                    if filename1 != prev:
                        cpt_train += 1
                        
                    if class_name in class_train_cpt:
                        class_train_cpt[class_name] += 1
                    else:
                        class_train_cpt[class_name] = 1
            else:
                    writer_test.writerow([newname,width, height,x1,y1,x2,y2,class_name])
                    if filename1 != prev:
                        cpt_test += 1
               
                    if class_name in class_test_cpt:
                        class_test_cpt[class_name] += 1
                    else:
                        class_test_cpt[class_name] = 1
            # save new data (multi channel) in .npz format            
            if filename1 != prev and convert_to_npz==True:
                img = None
                for img_name in filename:
                    temp = cv2.imread(img_name, 0)
                    if temp is None:
                        print("can't load image :", img_name)
                        break
                    temp = np.expand_dims(temp, axis=-1)
                    if img is None:
                        img = temp
                    else:
                        img = np.concatenate([img, temp], axis=-1)
                
                np.savez_compressed(directory+newname, img)
                count += 1
            prev = filename1
                    
                
                
                
                
                
            
            if class_name not in class_mapping:
                if class_name == 'bg' and found_bg == False:
                    print('Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
                    found_bg = True
                class_mapping[class_name] = len(class_mapping)  
    for class_, idx_ in class_mapping.items():
        writer_class.writerow([class_, idx_])
    print('number image training :', cpt_train)
    print('number image test :', cpt_test)
    print('count : ', count)
    print('number of label in train :', class_train_cpt)
    print('number of label in test :', class_test_cpt)    

parser = OptionParser()
parser.add_option('-f', "--filename", dest="filename", help='.txt filename to convert in .csv format (retinanet)', default='raccoon.txt')        
parser.add_option('-d', "--dir", dest="dir", help='directory where images are', default='')
parser.add_option('-c', "--channels",  help='number of channels', type=int,default=3)
parser.add_option('-n', '--newname', help='name of the new data', default='new')
parser.add_option('--npz', help="convert to .npz format", action="store_true")
(options, args) = parser.parse_args()

ConvertToRetinaNetFormatMultiCSV('labels/raccoon.txt', directory='', new=options.newname,channels=options.channels, convert_to_npz=options.npz)

