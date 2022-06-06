"""
###############################################################
#Import for neural network
###############################################################
"""
from keras import backend as K
from keras.models import load_model
from keras.models import model_from_json
import json
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
# Import necessary components for face detection
import numpy as np
import pandas as pd
import cv2
import dlib
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior() 
# Import necessary components for creation of xml file
from xml.etree import ElementTree, cElementTree
from xml.dom import minidom
import xml.etree.ElementTree as etree
import xml.etree.ElementTree as et
from xml.etree.ElementTree import tostring
"""
###############################################################
#FUNCTIONS
###############################################################
"""
def get_landmarks(im):
    im = np.array(im, dtype='uint8')
    faces = cascade.detectMultiScale(im, 1.15,  4, 0, (100, 100))
    if (faces==()):
        return np.matrix([[0 for row in range(0,2)] for col in range(Indicesface)])        
    else:
        for (x,y,w,h) in faces:
            rect=dlib.rectangle(int(x),int(y),int(x+w),int(y+h))
        return np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])
##############################################################################
##############################################################################    
def crop_face(im):
    faces = cascade.detectMultiScale(im, 1.15,  4, 0, (100, 100))
    if (faces==()):
        l=np.matrix([[0 for row in range(0,2)] for col in range(Indicesface)])
        rect=dlib.rectangle(0,0,0,0)
        return np.empty(im.shape)*0,l
    else:
        for (x,y,w,h) in faces:
            rect=dlib.rectangle(int(x),int(y),int(x+w),int(y+h))
            l=np.array([[p.x, p.y] for p in predictor(im, rect).parts()],dtype=np.float32)
            sub_face = im[y:y+h, x:x+w]
        return sub_face ,l 
##############################################################################
##############################################################################
def annotate_landmarks(im, landmarks):
    img = im.copy()
    if (landmarks.all()==0):
        return im
    else:
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            cv2.circle(img, pos, 2, color=(255, 255, 255), thickness=-1)
        return img
##############################################################################
##############################################################################
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
##############################################################################
##############################################################################
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
##############################################################################
##############################################################################
def fbeta_score(y_true, y_pred, beta=1):
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score
##############################################################################
##############################################################################
def fmeasure(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=1)

"""
###############################################################
#PARAMETERS
###############################################################
"""
PREDICTOR_PATH = '.../Experimento_final/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade_path='.../Experimento_final/haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)
Indicesface=68
im_s = 96
"""
###############################################################
#Model
###############################################################
"""
json_file = open('.../Experimento_final/squeezenet_u_final.json', 'r')
modelu_json = json_file.read()
modelu = model_from_json(modelu_json)
#modelu.load_weights('.../Experimento_final/squeezenet_u_final.h5',custom_objects={'fmeasure': fmeasure, 'precision':precision, 'recall':recall})
modelu.load_weights('.../Experimento_final/squeezenet_u_final.h5')
modelu.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy',fmeasure, precision, recall])

json_file = open('.../Experimento_final/squeezenet_l_corpus_.json', 'r')
modell_json = json_file.read()
modell = model_from_json(modell_json)
#modelu.load_weights('.../Experimento_final/squeezenet_l_corpus.h5',custom_objects={'fmeasure': fmeasure, 'precision':precision, 'recall':recall})
modell.load_weights('.../Experimento_final/squeezenet_l_corpus.h5')
modell.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy',fmeasure, precision, recall])

#x_u = np.load(".../Experimento_final/data_annotations/x_u_disfa.npy")
x_u_train1 = np.load('.../Experimento_final/data_annotations/x_u_HM.npy')
x_u_train2 = np.load('.../Experimento_final/data_annotations/x_u_silfa.npy')
#
y_u = np.load(".../Experimento_final/data_annotations/y_u_disfa.npy")
y_u_train1 = np.load('.../Experimento_final/data_annotations/y_u_HM.npy')
y_u_train2 = np.load('.../Experimento_final/data_annotations/y_u_silfa.npy')
##############################################################################
##############################################################################
#X_u=np.append(np.append(x_u,x_u_train1),x_u_train2)
Y_u=np.append(np.append(y_u,y_u_train1[:int(x_u_train1.size/(60*97*1))]),y_u_train2[:int(x_u_train2.size/(60*97*1))])
Y_u=np.nan_to_num(Y_u)
#Y_u=Y_u.astype('float32')
#X_u=np.nan_to_num(X_u)
#X_u = X_u.astype('float32')
##############################################################################
##############################################################################
img_rows_u, img_cols_u = 60, 97
#X_u = X_u.reshape(int(X_u.shape[0]/(60*97*1)), img_rows_u, img_cols_u, 1)
input_shape = (img_rows_u, img_cols_u, 1)
##############################################################################
##############################################################################
# convert class vectors to binary class matrices
encoder_u = LabelEncoder()
encoder_u.fit(Y_u)
encoded_Y_u = encoder_u.transform(Y_u)
# convert integers to dummy variables (i.e. one hot encoded)
Y_u = np_utils.to_categorical(encoded_Y_u)
num_classes_u = Y_u.shape[1]
print(num_classes_u)
labels_encoded_u=encoder_u.inverse_transform(encoded_Y_u)
labels_ordered_u=np.sort(labels_encoded_u)
labels_ordered_u=np.append(labels_ordered_u,74)
labels_ordered_u=set(labels_ordered_u)
labels_ordered_u=np.fromiter(labels_ordered_u, int, len(labels_ordered_u))
##############################################################################
##############################################################################
#x_l = np.load('.../data_annotations/x_l_disfa.npy')
x_l_train1 = np.load('.../Experimento_final/Data_annotations/x_l_HM.npy')
x_l_train2 = np.load('.../Experimento_final/Data_annotations/x_l_silfa.npy')
#
y_l = np.load('.../data_annotations/y_l_disfa.npy')
y_l_train1 = np.load('.../Experimento_final/data_annotations/y_l_HM.npy')
y_l_train2 = np.load('.../Experimento_final/data_annotations/y_l_silfa.npy')
##############################################################################
##############################################################################
#X_l=np.append(np.append(x_l,x_l_train1),x_l_train2)
Y_l=np.append(np.append(y_l,y_l_train1[:int(x_l_train1.size/(36*98*1))]),y_l_train2[:int(x_l_train2.size/(36*98*1))])
#X_l=np.nan_to_num(X_l)
Y_l=np.nan_to_num(Y_l)
#Y_l=Y_l.astype('float32')
#X_l = X_l.astype('float32')
##############################################################################
##############################################################################
img_rows_l, img_cols_l = 36, 98
#X_l = X_l.reshape(int(X_l.shape[0]/(36*98*1)), img_rows_l, img_cols_l, 1)
input_shape = (img_rows_l, img_cols_l, 1)
##############################################################################
##############################################################################
# convert class vectors to binary class matrices
encoder_l = LabelEncoder()
encoder_l.fit(Y_l)
encoded_Y_l = encoder_l.transform(Y_l)
# convert integers to dummy variables (i.e. one hot encoded)
Y_l = np_utils.to_categorical(encoded_Y_l)
num_classes_l = Y_l.shape[1]
print(num_classes_l)
labels_encoded_l=encoder_l.inverse_transform(encoded_Y_l)
labels_ordered_l=np.sort(labels_encoded_l)
labels_ordered_l=np.append(labels_ordered_l,73)
labels_ordered_l=set(labels_ordered_l)
labels_ordered_l=np.fromiter(labels_ordered_l, np.int64, len(labels_ordered_l))
#print(labels_ordered_l)
##############################################################################
##############################################################################
labels_u=[        '0',         '1',         '2',         '4' ,       ' 5 ',        '6',        ' 9',
'        10   ' ,   ' 12    ',   ' 13   ',     '14',        '15',        '16',        '19',
'      61+63  ' ,   '  23   ',   '  2+4 ',      ' 2+5',        '2+6',        '35',        '41',
'        42   ' ,   ' 43    ',   ' 44   ',     '45',        '46 ',       '4+9',       '5+62',
'       5+63  ' ,   ' 5+64  ',   '   56 ',      ' 57',        '58 ',      '5+72',        '61',
'        62   ' ,   ' 63    ',   ' 64   ',    '5+73',      '4+6+72',      '41+63',      '41+64',
'        69   ' ,   ' 70    ',   ' 71   ',     '72',        '73 ',       '74',   '4+43+70+71',
'  1+2+43+70+71',    ' 4+61+63',  '  1+2+70+71',    '44+70+71',      '62+63',      '62+64',       '1+2+4',
'       1+2+5  ',   '  1+2+6 ',   '   1+2+9' ,      '1+42',       '1+43',       '1+46',       '1+4+9',
'      42+61   ',   '42+62    ',  '42+63   ' ,  '42+70 ',     '42+72',      '42+73',     '4+62+64',
'      1+2+31  ',   '4+42+44  ',  '  1+2+41' ,     '1+2+42 ',     '1+2+43',      '1+2+44',      '1+2+46',
'      1+2+4+9 ',   ' 4+42+62 ',  '   1+2+57',      '1+2+58',      '1+2+61',      '1+2+62',      '1+2+63',
'      1+2+64  ',   '4+42+73  ',  '   2+42  ',   '5+70+71',     '1+2+5+32',       '2+46',      '1+2+70',
'      1+2+71  ',   '  2+4+9 ' ,  '  1+2+72 ',     '1+2+73',       '2+5+6',    '1+2+41+61',    '1+2+41+62',
'      43+61   ',   '43+63   ' ,  '43+64    ', ' 43+70',      '43+72',      '43+73',     '1+2+5+71',
'     1+2+5+72 ',   ' 4+43+44' , '1+2+42+70+71',     '4+43+58',    '  44+31',      '4+9+43',      '4+9+44',
'     4+43+72  ',   '4+43+73 ' ,  '42+70+71 ',  '4+44+70+71',     ' 4+9+63', '4+44+46+70+71',     '4+44+46',
'      70+71   ', ' 4+44+58  ' ,  '4+44+61  ',  ' 4+44+62 ',      '4+31',       '4+32',       '9+44',
'   4+42+61+63 ', '   4+44+72' ,  '    4+41 ',   '   4+42',       '4+43',       '4+44',     '4+44+73',
'    1+2+43+57 ',  '    4+61 ' ,  '   4+62  ',   '  4+63',       '4+64',    '1+2+43+72',       '4+6+9',
'       4+71   ',  '  4+72   ' ,  ' 4+73    ', '4+70+71']
print('labels_upper=',len(labels_u))
labels_l=['0',              '1',              '2',           '20+51',              '4',
  '            5',           '20+52'    , '20+25+26+51+53',             '10' ,            '12',
   '          13 ',            '14'     ,       ' 15'     ,        '16'       ,      '17',
    '         18 ',       '19+25+26'   ,         ' 20'    ,     '19+25+28'  ,           '22',
     '        23 ',            '24'     ,        '25  '   ,        '26',             '27',
      '       28 ',         '5+12+25'   ,    '12+25+52+53',       '12+25+52+56 ',      '19+22+25+51',
   '22+25+26+52+54+72',             '34' ,           ' 35 ',       ' 19+25+51  ',       '19+25+52',
         '19+25+53',       '22+25+56+57'  ,   '15+16+25+52+72',     '19+25+26+54+56',     '18+22+25+51+53',
          '   51',             '52'        ,     '53' ,            '54',             '55',
           '  56 ',            '57'        , '23+55+72',       '22+25+56+72',             '62',
         '15+16+17',     '18+22+25+51+72'  ,        ' 21+17',             '72',             '73',
         '15+16+25',   '15+16+25+53+55+72' ,        '13+52+53',         '13+52+54',       '12+22+25+51',
       '23+52+54+72',         '15+16+51'   ,    '23+34+52+54 ',        '15+16+55',     '12+25+26+52+56',
          '5+33+53',     '20+25+26+52+54'  ,     ' 5+33+51+53',     '15+16+25+53+55',   '    12+25+53+56',
            '1+25 ',        '10+25+26'     ,  ' 5+22+25+26',     '12+25+52+53+72'    , '10+22+25+52+53',
     '18+22+25+52+53',         '10+25+51'  ,  ' 18+22+25+52+54',        '5+22+25+53' ,  '  18+22+25+52+55',
      '  5+22+25+56',         '15+17+20'   ,  '18+22+25+52+72',     '18+22+25+52+73' ,   '     15+17+24',
       '  15+17+25 ',        '15+17+26'    ,  '     22+25' ,        '17+22+25' ,      '20+26+51+53',
        ' 15+17+51 ',        '15+17+52'    ,  '   15+17+53 ',       ' 15+17+54',       '    22+51',
         '15+17+55 ',          '22+53'     ,  '  15+17+56 ' ,       '  22+55'  ,   '20+25+26+53+55',
         '15+17+72 ',      '15+20+25+26'   ,  '  12+25+54+55',       '12+25+54+56',           ' 2+25',
     '25+26+52+56+72',       '15+20+25+53' ,  '         2+51 ' ,      ' 23+57+72',            '2+53',
     '18+22+25+53+55',     '18+22+25+53+72',  '   15+20+25+52+53',     '      23+26',         '18+25+54',
      '     23+34   ',    '34+52+54+72'    ,  '     23+51',           '23+52',           '23+53',
       '    23+54   ',        '23+55'      ,  '   23+56'  ,         '23+57'  ,       '17+23+51',
        ' 17+23+54   ',        '23+72'     ,  '  12+52+53',         '12+52+54',       '16+22+25+51',
       '16+22+25+52  ',     '15+20+26+53'  ,  '   10+16+25+26',     '12+17+20+25+26',           ' 3+55',
       '10+25+26+51  ',     '10+25+26+52'  ,  '   10+25+26+53 ',    '15+17+52+53+56',   '15+17+25+52+53+56',
       '10+25+26+56 ',        '15+19+25'   ,  '22+25+27+52+53 ' ,     '10+16+25+52',       '10+16+25+53',
       '    24+26 ',      '10+16+25+55'    ,  ' 20+26+53+55' ,          '24+51',           '24+52',
       '    24+53 ',          '24+54'      ,  '   24+55'  ,         '24+56' ,          '24+57',
       '10+12+16+25',       '15+16+17+52'  ,  ' 19+22+25+52+53',         '17+24+51',          ' 24+72',
       '  16+22+25 ',     '5+15+16+25+51'  ,  '   12+25+56+72',       '15+17+20+25',       '15+17+20+26',
       '10+25+27+52',     '15+17+52+54+56' ,  '15+17+25+52+54+56',         '15+20+25',         '15+20+26',
       '    25+26  ',         '25+27'      ,  '   25+28',         '28+51+53',       '19+20+25+27',
       '  28+51+55 ',          '25+32'     ,  '15+17+20+73',     '18+20+25+26+51',     '26+20+25+26+51',
       '  17+25+26 ',        '15+20+51'    ,  '   28+51+72',           '25+51',           '25+52',
       '    25+53  ',         '25+54'      ,  '   25+55',           '25+56',           '25+57',
       ' 5+12+25+51',        '5+12+25+52'  ,  '    5+12+25+56',     '19+22+25+53+55',     '22+25+26+51+53',
     '22+25+26+51+54',     '22+25+26+51+55',  '         25+72',      '      5+25',           ' 5+26',
     '    12+54+55   ',      '12+54+56'    ,  ' 12+20+25+26',       '24+26+52+53',            '5+53',
     '18+22+25+56+57',       '15+17+51+55',   '  15+16+25+52+54',     '12+22+25+51+53',     '15+16+20+25+51',
     '      26+28',         '28+52+53'   ,    '  28+52+54 ',          '26+33',       '13+25+52+53',
     '15+17+25+52+53',     '15+17+25+52+54' , '   15+17+25+52+56',      '     26+51' ,     ' 34+51+55+57',
     '22+25+26+52+53',     '22+25+26+52+54',  '   22+25+26+52+55',     '22+25+26+52+56',    ' 22+25+26+52+58',
     '  14+25+26+51 ',      '14+25+26+52'  ,  '   14+25+26+55',       '14+25+26+56',         '28+32+53',
     '22+25+26+52+72',       '23+51+57+72' ,  '    25+28+52+53',     '    25+25+51' ,       '  5+18+52',
     '  24+52+54+56 ',        '15+22+25'  , '15+17+19+25+26+52',    '     12+15+17'  ,   '10+16+25+51+53',
     ' 5+18+22+25+26',         '26+28+52' ,  '    15+16+20+52 ',    '10+16+25+51+72',    '     26+28+56',
     '    25+26+28 ' ,        '5+19+25'   ,   '   16+25+25',        ' 16+25+26',        ' 25+26+51',
     '    25+26+52 ' ,       '25+26+53'   ,  '    25+26+54',       '  25+26+55' ,       ' 25+26+56',
     '    25+26+57 ' ,     '25+28+53+55'  ,    '   16+25+51',     '    16+25+52' ,      '12+17+20+26',
     '    16+25+55 ' ,       '25+26+72'   ,  '15+25+26+51+53' ,  '15+19+25+26+51+53',   '    25+26+28+51',
     '  25+26+28+52 ',      '25+26+28+53' ,  '    20+24+52+54',     '  25+26+28+54' ,    '  25+26+28+55',
     '      28+25  ' ,    '25+26+28+56'   ,  '15+25+26+51+57' ,     '   16+25+72' ,    '15+16+19+25+26',
     '    28+54+56 ' ,        '1+10+25'   ,   '  5+18+22+25'  ,     '25+27+51+53',        ' 12+16+25',
     '  16+20+25+26' ,        '15+23+53'  ,   '    28+54+73'  ,     '    28+51' ,         ' 28+52',
     '      28+53  ' ,        '28+54'     ,   '   28+55'      ,     '28+56',       '12+19+25+51',
     '  25+27+51+72' ,      '12+19+25+55' ,   '   25+27+51+73',     '22+25+26+54+56',   '    10+25+51+53',
     '  16+20+25+51' ,      '16+20+25+52' ,   ' 10+12+16+25+72',      '   18+51+53',     '    18+51+54',
     '    18+51+56 ' ,      '5+19+25+26'  ,   '    25+27+51'   ,     ' 25+27+52' ,        '25+27+53',
     '    25+27+54 ' ,   '25+26+28+51+53' ,   '     25+27+56'  ,     '  35+51+57',     '15+25+26+52+53',
     '  15+17+24+55' ,      '24+52+56+72' ,   ' 15+25+26+52+56',     '  18+25+26+51',   '    25+27+52+53',
     '  25+27+52+54 ' ,     '25+27+52+55' ,   '     12+17+20 ' ,    ' 25+27+52+56',    ' 10+16+25+53+55',
     '10+16+25+53+56' ,        '12+17+25' , '       15+24+54 ' ,    ' 25+27+52+72',     '   5+15+16+25',
     '  10+25+52+53 ',        '18+52+53'  ,  '     18+52+54'   ,    '15+17+25+26' ,      '  25+28+51',
     '    25+28+55  ',       '18+52+72'   ,   '   18+52+73 '   ,     '25+28+56'   ,    '15+17+25+52',
     '  15+17+25+55 ',        '15+25+26'  ,    ' 13+23+51+56'  ,     '10+22+25+51',     '  10+22+25+52',
     '    24+26+51  ',       '24+26+52'   ,    '25+27+53+55 '  ,     ' 24+26+55'  ,      ' 15+25+51',
     '    15+25+52  ',       '15+25+53'   ,    '25+27+53+72 '  ,    '10+25+53+56' ,       '   10+25',
     '     5+22+25  ',     '25+26+51+53'  ,    ' 25+26+51+54'  ,     '25+26+51+57',        ' 18+53+55',
   '15+25+26+52+54+56',         '53+55+72',    '   25+26+51+72',           '10+52'   ,      '18+53+73',
   '      18+53+72   ',        '51+53'   ,     '   51+54'   ,  '22+25+51+55+57'      ,   '17+51+53',
   '        51+72    ',     '17+51+57'   ,    '13+23+52+53' ,      '13+23+52+56'     ,   ' 52+53+72',
   '      34+51+55   ',      '15+26+51'  ,    '   15+26+54' ,        '  72+51'       ,  '34+51+72',
   '      34+51+73   ',  '10+16+25+55+72',    ' 15+17+20+25+26',    '   25+26+52+53' ,   '      5+23+26',
   '    25+26+52+54  ',     '25+26+52+55',    '   25+26+52+56',    ' 16+25+26+52+53' ,    '  22+25+26+51',
   '    22+25+26+52  ',     '22+25+26+53',    '   25+52+53+55',   '10+16+25+52+53+55',     '  22+25+26+56',
   '    22+25+26+57   ',    '25+26+52+72',    '   25+26+52+73',      ' 25+52+53+72'  ,   '16+25+52+53+72',
   '10+16+25+52+53+72 ',         '5+23+51' ,  '12+16+25+52+53+72',          '5+23+53',    '   22+25+26+73',
   '        52+53     ',      '52+54'     ,   '   52+57',   '15+18+22+25+52+53 '  ,       ' 52+72',
   '      17+52+57    ',   '15+16+25+26'  ,   '    52+54+72',    '     52+54+73'  ,     '35+51+55+57',
   '      12+20+25    ',   '15+25+26+51 ' ,   '  15+25+26+52',   '    15+25+26+53',     '  15+25+26+54',
   '    15+25+26+55   ',    '18+52+53+55' ,   '     25+51+53' ,  '      25+51+54 ',      '  25+51+55',
   '      25+51+56    ',   '15+16+25+51'  ,   '    34+52+72' ,   '   15+16+25+52 ',      '15+16+25+53',
   '    15+16+25+55   ',      '14+25+26'  ,   '      12+15 ' ,   '      12+17'  ,        ' 12+20',
   '      25+51+72',   '10+12+16+25+51+72',   '        12+23',   '        12+24',        '   12+25',
   '        12+26',         '23+26+51 '  ,    '    12+28'  ,     '25+26+53+55 ' ,     '25+26+53+56',
   '    22+25+27+51',       '22+25+27+52',     '  17+23+51+53',    '   22+25+27+53',   '    25+52+54+56',
   '      14+25+51',         '14+25+52'  ,     '  14+25+55'   ,    '   2+17+26 ',       '   12+51',
   '        12+52',           '12+53'    ,     ' 5+24+52 '    ,    '  12+55 '   ,       '12+56',
   '        53+55',           '53+56 '   ,   '12+17+25+26'    ,    ' 10+16+25'  ,        ' 53+72',
   '    15+18+51+54',         '13+23+56' ,   '     25+52+53'  ,    '   25+52+54',         '  33+51',
   '      25+52+56',           '33+53  ' ,   '     13+14   '  ,   ' 25+52+72 '  ,        '13+23',
   '      51+53+72 ',         '5+25+26 ' ,   '     5+25+27 '  ,   ' 25+26+54+56 ',    '15+22+25+51+54',
   '      33+51+56 ',     '17+23+52+53' ,    '    18+56+57 '  ,   '   22+25+26' ,      '  22+25+27',
   '        13+51  ',         '13+52 '   ,    '   13+53    '  ,   ' 5+25+52 '   ,       '13+56',
   '        54+55   ',    '12+25+26+53'  ,    '   22+25+51 '  ,     ' 22+25+52 ',        '22+25+53',
   '      22+25+54  ',       '22+25+55'  ,    '   22+25+56 '  ,     ' 22+25+57' ,      '17+20+25+26',
    '   17+20+25+27 ',          '54+72'  ,    ' 17+20+25+28'  ,     '17+20+25+29',      ' 17+20+25+30',
    '   17+20+25+31 ',      '17+20+25+32',    '   17+20+25+33',       '17+20+25+34',     '    52+56+72',
    '   13+19+25+52 ',      '17+20+25+35',    '   17+20+25+36 ',      '  22+25+72',   '10+16+25+51+53+72',
    '   25+51+53+55 ',          '34+51'  ,    '     34+52'     ,     ' 34+53' ,        '  34+54',
    '       34+55   ',      '25+53+55 '  ,    '  25+53+56'     ,    '51+54+55',         '25+53+72',
    '     17+34+55  ',         '14+23'   ,    '    34+72 '     ,   '  14+25',           '34+73',
    '     51+54+72  ',       '33+52+53'  ,    ' 19+25+26+28'   ,  '15+22+25+52+53',     '    24+51+53',
    '     24+51+54  ',       '24+51+55'  ,    '   24+51+56 '   ,    ' 24+51+57'   ,    '25+52+56+72',
    ' 15+22+25+52+72',           '14+51' ,    '      14+52 '   ,   '19+25+26+51'  ,    ' 19+25+26+52',
    '   19+25+26+55 ',          '14+56'  ,   '16+19+25+26+52' ,   '   18+22+25+26',    '   25+51+54+55',
    '   19+25+26+72 ',          '55+72'  ,    '     55+73' ,    '25+26+28+52+53 ' ,    '   13+25+51',
    '     13+25+52  ',     '20+15+17+52' ,    '10+15+17+25+51' ,    '25+26+28+52+55',  '   20+15+17+20+52',
    '   18+22+25+51 ',      '18+22+25+52',    '   18+22+25+53' ,     ' 18+22+25+54',   '    18+22+25+55',
    '   18+22+25+56 ',          '35+51'  ,    '     35+52',         '25+54+55'   ,  '12+16+25+54+56',
     '  12+25+26+51 ',          '15+16'  ,    '     15+17',       '12+25+26+52' ,   '     12+23+51',
    '       15+20   ',      '12+23+52'   ,    '    15+22 ',          '15+23'    ,    '   15+24',
    '       15+25   ',        '15+26'    ,    ' 51+55+72' ,      '12+25+26+56'  ,    ' 12+16+25+53',
    '   15+17+51+53 ',      '15+17+51+54',    '   12+16+25+56',     '    24+52+53',  '       24+52+54',
    '     24+52+55  ',       '24+52+56'  ,    '   24+52+57' ,      '16+17+25+52',    '5+22+25+26+51+57',
    '   12+16+25+72 ',      '18+22+25+72',    '       15+51',          ' 15+52' ,    '      15+53',
    '       15+54   ',        '15+55'    ,    '   15+56' ,      '18+22+25+73',       '19+25+27+52',
    ' 13+16+25+52+54',         '24+52+72',    ' 19+25+27+51+53',     '19+25+27+51+54',         '15+51+72',
    '       56+72   ',  '25+26+28+53+55' ,       '5+25+26+53'  ,    ' 22+25+51+53' ,      '22+25+51+54',
    '   22+25+51+55 ' ,     '22+25+51+56',       '22+25+51+57' ,     '  5+16+25+52',      ' 22+25+51+72',
    '   22+25+51+73 ',          '16+17'  ,       '12+24+51'    ,    ' 12+24+52'  ,       '25+55+72',
    '   15+17+52+53 ',          '16+23' ,       ' 12+24+55'    ,      ' 16+25'   ,        '16+26',
   '    15+17+52+54 ',      '15+17+52+56',     '10+25+26+52+53',     '    24+53+55',     '18+22+25+26+51',
    ' 18+22+25+26+52',     '18+22+25+26+53',  '   18+22+25+26+56',  '     15+25+51+53',   '    15+25+51+54',
    '       16+51   ',  '19+25+27+52+51' ,    '19+25+27+52+53',    '   19+25+28+55',     '   5+25+27+51',
    '     23+51+53  ',       '23+51+54'  ,     ' 5+25+27+53'  ,   '    23+51+57'   ,    '22+25+52+53',
    '   22+25+52+54 ',      '22+25+52+55',     '  22+25+52+56',  '     22+25+52+57',      '   12+25+26',
    '     23+51+72  ',     '22+25+52+72',     '    25+56+57' ,   '   22+25+52+73' ,      '12+25+28+52',
    '     12+25+51  ',         '17+20'  ,    '   12+25+52'   ,   '   12+25+53'   ,      '  17+23',
    '     12+25+55  ',         '17+25'  ,    '     17+26'    ,   '  12+25+56'    ,     '24+54+56',
    '   15+25+52+53 ',          '17+51' ,    '      17+52'   ,   '     17+53'    ,       '17+55',
    '       17+57   ',      '19+22+25 ' ,    ' 16+25+26+51'  ,   '      17+72'   ,    '16+25+26+52',
    '   16+25+26+55 ',        '23+52+53',    '     23+52+54' ,   '     23+52+55 ',        '23+52+56',
    '     23+52+57  ',     '22+25+53+55',    ' 19+25+26+51+54',  '   19+25+26+51+56',         '25+16+53',
     '    23+52+72  ',     '15+22+25+51',    '   15+22+25+52',   '    15+22+25+53',       '15+22+25+55',
    '       18+22   ',      '20+25+27'  ,    '   20+25+26'   ,   '     18+26'  ,       '20+25+32',
    '    5+15+25+52   ',        '18+34'   ,  '15+16+17+51+53',   '      20+25+51',         '20+25+52',
    '     20+25+53  ',         '18+51'  ,     '    18+52'   ,    '    18+53 ',          '18+54',
    '       18+55   ',        '18+56  ' ,     '   18+57'    ,    '2+32+63+51',       '25+31+51+53',
    '     23+53+55  ',       '23+53+56 ',     ' 22+25+54+56',     '19+25+26+52+53 ',  '10+19+25+26+52+53',
 '10+16+19+25+26+52+53',         '14+52+56',  '   25+26+28+56+73', '    19+22+25+26+51',     '19+22+25+26+52',
  '     19+25+51+53 ',         '5+51+72 ' ,   '      19+25'    ,   '    19+28 ' ,       '20+26+51',
   '      20+26+52  ',     '18+20+25+26 ' ,   '25+27+28+51+54' ,   ' 25+26+52+53+55',     '25+26+52+53+56',
    '   12+25+51+53 ',      '12+25+51+54' ,    '25+26+52+53+72',   '22+25+26+52+53+73',         '10+23+52',
     '    18+22+25  ',       '18+22+26 ' ,     '23+26+52+53'   ,    '    20+24 ' ,         '20+25',
      '     20+26   ',      '23+34+51  ' ,      '18+22+51'    ,     '18+22+52 '  ,        '20+27',
      ' 20+25+26+51 ',      '20+25+26+52',     '    13+51+53',         '23+34+57',     '10+12+16+25+51',
      '   17+20+25  ',       '17+20+26',       '23+52+53+72',       '23+52+53+73']
print('labels_lower=',len(labels_l))
##############################################################################
##############################################################################
def neural_net(path):
    v_entry=cv2.VideoCapture(path,0)
    Frames = int(v_entry.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = v_entry.get(cv2.CAP_PROP_FPS)
    output_u=[]
    output_l=[]
    M=[]
    for i, j in enumerate(range(0,Frames)):
        v_entry.set(1, int(j))
        ret, im = v_entry.read()
        points_u=np.empty((21,2))*0
        points_l=np.empty((32,2))*0
        if ret is True:
            a,l = crop_face(im)
            c=get_landmarks(a)
            points_u[:9,:]=c[17:26,:]
            points_u[10:,:]=c[36:47,:]
            vp=np.stack((points_u))
            points_l[:12,:]=c[2:14,:]
            points_l[13:,:]=c[48:67,:]
            vb=np.stack((points_l))
            vs_brown_e=np.squeeze(np.asarray(c[19]-c[17]))
            vi_brown_e=np.squeeze(np.asarray(c[21]-c[17]))
            vs_brown_d=np.squeeze(np.asarray(c[24]-c[26]))
            vi_brown_d=np.squeeze(np.asarray(c[22]-c[26]))
            a_brown_e=np.arccos(np.dot(vs_brown_e,vi_brown_e,out=None)/(np.linalg.norm(vs_brown_e)*np.linalg.norm(vi_brown_e)))
            a_brown_d=np.arccos(np.dot(vs_brown_d,vi_brown_d,out=None)/(np.linalg.norm(vs_brown_d)*np.linalg.norm(vi_brown_d)))
            v1_eye_e=np.squeeze(np.asarray(c[37]-c[41]))
            v2_eye_e=np.squeeze(np.asarray(c[38]-c[40]))
            v1_eye_d=np.squeeze(np.asarray(c[43]-c[47]))
            v2_eye_d=np.squeeze(np.asarray(c[44]-c[46]))
            vs=np.stack((vs_brown_e,vi_brown_e,vs_brown_d,vi_brown_d,v1_eye_e,v2_eye_e,v1_eye_d,v2_eye_d))
            d_lips_h1=np.squeeze(np.asarray(c[48]-c[54]))
            d_lips_h2=np.squeeze(np.asarray(c[60]-c[64]))
            d_lips_v1=np.squeeze(np.asarray(c[51]-c[57]))
            d_lips_v2=np.squeeze(np.asarray(c[62]-c[66]))
            vl=np.stack((d_lips_h1,d_lips_h2,d_lips_v1,d_lips_v2))
            p_u=[vp.tolist(), vs.tolist()]
            points_upper=np.hstack([np.hstack(np.vstack(p_u)),a_brown_e,a_brown_d])
            p_l=[vb.tolist(), vl.tolist()]
            points_lower=np.hstack(np.vstack(p_l)).reshape((36,2))
            r = cv2.resize(a, dsize=(im_s, im_s), interpolation=cv2.INTER_CUBIC)
            r = r[:,:,1]
            upper = np.array(r[:60,:])
            lower = np.array(r[60:,:])
            im_u = np.vstack((upper.T,points_upper))  
            im_u = im_u.astype('float32')
            im_u /= 255
            im_l = np.vstack((lower.T,points_lower[:,0],points_lower[:,1]))
            im_l = im_l.astype('float32')
            im_l /= 255
            x_upper = np.expand_dims(im_u, axis=0)
            x_lower = np.expand_dims(im_l, axis=0)
            x_upper=x_upper.reshape((1, 60, 97, 1))
            x_lower=x_lower.reshape((1, 36, 98, 1))
            exit_u = modelu.predict(x_upper)
            exit_l = modell.predict(x_lower)
            exit_u=np.argmax(exit_u, axis=1)
            exit_l=np.argmax(exit_l, axis=1)
            e_labels_u=encoder_u.inverse_transform(exit_u)
            e_labels_l=encoder_l.inverse_transform(exit_l)
            print(e_labels_u)
            print(e_labels_l)
            output_u = np.append(output_u, e_labels_u)
            output_l = np.append(output_l, e_labels_l)
        else:
            output_u = np.append(output_u,74)
            output_l = np.append(output_l, 72)
            continue
    
    all_exit_u=np.matrix(zip(range(0,Frames),output_u))
    all_exit_l=np.matrix(zip(range(0,Frames),output_l))
    
    root = et.Element('TIERS', **{'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance'}, **{'xsi:noNamespaceSchemaLocation': 'file:avatech-tiers.xsd'})
    somedata = et.SubElement(root, 'TIER', columns="AUs") 
    for m,n in enumerate(range(0,Frames)):
        print(m)
        if (np.where(labels_ordered_u==output_u[m])):
            a=np.where(labels_ordered_u==output_u[m])
            print(a)
            print(labels_u[int(a[0][0])])
            if (np.where(labels_ordered_l==output_l[m])):
                b=np.where(labels_ordered_l==output_l[m])
                print(b)
                print(labels_l[int(b[0][0])])
                ms_inicial=round((m*(1000 / (fps / 1.001)))*.001,3)
                ms_final=round(((m+1)*(1000 / (fps / 1.001)))*.001,3)
                full_elan_exit_u=("<span start= \"%s\" end=\"%s\" ><v>%s</v></span>"%(ms_inicial,ms_final,labels_u[int(a[0][0])]))
                child1 = ElementTree.SubElement(somedata,"span", start='%s'%(ms_inicial), end="%s"%(ms_final))
                v = etree.Element("v")
                v.text = "%s+%s"%(labels_u[int(a[0][0])],labels_l[int(b[0][0])])
                child1.append(v)
                tree = cElementTree.ElementTree(root) # wrap it in an ElementTree instance, and save as XML
                t = minidom.parseString(ElementTree.tostring(root)).toprettyxml() # Since ElementTree write() has no pretty printing support, used minidom to beautify the xml.
                tree1 = ElementTree.ElementTree(ElementTree.fromstring(t))
                print(tree1)
            else:
                continue
        else:
            continue
        
    return tree1            
#call the network function
my_address='.../Video_test/bomdia_libras.mp4'
output=neural_net(my_address)
#output.write("video_test_libras.xml",encoding="utf-8", xml_declaration=True)
