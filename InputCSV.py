# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 15:27:15 2018

@author: Z930
"""
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
# from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

import csv
import datetime
import os
import re

# from datetime import time
# import time


class DataSet(object):
  m_list_TrainingDataBy_ClipLength = [];
  ClipLength = 16;
  InputItemsLength = 24 ;
  
  def Shuffel_and_Reassign_Dataset():
      
      print("\n shuffle and reassign dataset \n");
      
      numpy.random.shuffle(DataSet.m_list_TrainingDataBy_ClipLength);
    
      train_Clips , train_Labels = separate_Clip_and_Label( DataSet.m_list_TrainingDataBy_ClipLength );
    
      shuffled_train_DataSet = DataSet(train_Clips ,train_Labels ,dtype=dtypes.float32 ,reshape=False)   
      
      
      
      return shuffled_train_DataSet ; 
      
  
  def __init__(self,
               clips,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=False):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert clips.shape[0] == labels.shape[0], (
          'clips.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = clips.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert clips.shape[3] == 1
        clips = clips.reshape(clips.shape[0],
                                clips.shape[1] * clips.shape[2])
        

        
    self._clips = clips
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  
  def getDictStrAngleRadius_listRowsOfFile(self):
    return self.m_dict_strAngleRadius_listRowsOfFile

  @property
  def clips(self):
    return self._clips

  @property
  def images(self):
    return self._clips

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._clips = self._clips[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._clips[start:end], self._labels[start:end]

#############################################################



def parsingStringToDatetime ( strSimpleTime ):
    list_strItem = strSimpleTime.split(':')
    iM = int(list_strItem[0])
    iSec = int(list_strItem[1])
    iMicroSec = int(list_strItem[2])
    dtResult = datetime.datetime(1900,1,1,12,iM,iSec,iMicroSec);
    
    return dtResult;


def parsingStringToSeconds ( strSimpleTime ):
    list_strItem = strSimpleTime.split(':')
    iM = int(list_strItem[0])
    iSec = int(list_strItem[1])
    iMicroSec = float(list_strItem[2]) * 0.001; 
    secResult = iM*60 + iSec + iMicroSec;
    
    return secResult;

def normalizeRssiStrToFloat ( strRssiItem ):
    normalizeRssi = 0.0;
    rawRssiValue = float(strRssiItem) ;
    if rawRssiValue == 127.0 :
        rawRssiValue = 0.0;
        normalizeRssi = 0.0;
    else :    
        # 假設rawRssiValue 不會低於過 -100，並且在 0-負數區間
        rawRssiValue = rawRssiValue if rawRssiValue>-100.0 else -100.0 ;
        normalizeRssi = (rawRssiValue + 100.0) / 100.0 ;
        
    return normalizeRssi;

def processFile_To_NormalizedList ( strFullFilePath ):
    f = open(strFullFilePath ,'r')
    
    strLabelRow = 'Rssi time'
    list_NormalizedRowsOfFile = []
    #list_EachRow_label_CosSin = []
    
    iRow = 0;
    for strItems_ofRow in csv.reader(f):
        
        iRow += 1 ;
        #print ('> %d \n' % iRow) ;
        
        if strItems_ofRow[0] == strLabelRow or strItems_ofRow[0] =='':
            #print("Row: %d : DO NOT THING \n");
            # DO NOT THING
            continue ;
        else:
            
            list_ItemsOfRow = [];
            for i in range(len(strItems_ofRow)) :
                
                if   i==0 : #Rssi time
                    #dtTmp1 = parsingStringToDatetime (strItems_ofRow[i]);
                    #secTime = parsingStringToSeconds (strItems_ofRow[i]);
                    #normalizeTime = secTime /  86400.0  ; # 86400 is total seconds = one day
                    #list_ItemsOfRow.append(normalizeTime)
                    strtime = strItems_ofRow[i]
                    list_ItemsOfRow.append(strtime)
                    
                elif i==1 : # Rssi
                    strRssiItem = strItems_ofRow[i];
                    #normalizeRssi = normalizeRssiStrToFloat( strRssiItem ) ;
                    list_ItemsOfRow.append (strRssiItem); 
                    
                # 跳過 i=2 'Beacon time'        
                elif i==3 : # Beacon Rssi
                    strbeaconRssiItem = strItems_ofRow[i];
                   # normalizeBeaconRssi = normalizeRssiStrToFloat ( strItems_ofRow[i] ) ;
                    list_ItemsOfRow.append ( strbeaconRssiItem ) ;
                    
                
                elif i==4 : # Beacon Accury ,range is 0.0 ?30.0
                   # BeaconAccury_Max = 30.0
                    strItem = strItems_ofRow[i];
                   # BeaconAccury = float(strItem) ;
                    #BeaconAccury = BeaconAccury if BeaconAccury<BeaconAccury_Max else BeaconAccury_Max ;
                    #normalizeBeaconAccury = (BeaconAccury_Max - BeaconAccury ) / BeaconAccury_Max ;
                    #list_ItemsOfRow.append ( normalizeBeaconAccury ) ;
                    list_ItemsOfRow.insert(17 , strItem);
                
                # 跳過 i=5 'Sensor time'        
                elif i==5 : # Compass_Cosine ,range is -1.0 +1.0
                    Compass_Cosine = float(strItems_ofRow[i]) ;
                    #list_ItemsOfRow.append ( Compass_Cosine ) ;
                    list_ItemsOfRow.insert(0 ,Compass_Cosine);
                    
                elif i==6 : # Compass_Sine ,range is -1.0 +1.0
                    Compass_Sine = float(strItems_ofRow[i]) ;
                    #list_ItemsOfRow.append ( Compass_Sine ) ;
                    list_ItemsOfRow.insert(1 ,Compass_Sine);
                    
 
                elif i==7 : # Pitch_Cosine ,range is -1.0 +1.0
                    Pitch_Cosine = float(strItems_ofRow[i]) ;
                    list_ItemsOfRow.append ( Pitch_Cosine ) ;
                       
                elif i==8 : # Pitch_Sine ,range is -1.0 +1.0
                    Pitch_Sine = float(strItems_ofRow[i]) ;
                    list_ItemsOfRow.append ( Pitch_Sine ) ;
                    
                elif i==9 : # Roll_Cosine ,range is -1.0 +1.0
                    Roll_Cosine = float(strItems_ofRow[i]) ;
                    list_ItemsOfRow.append(Roll_Cosine);
                    
                elif i==10 : # Roll_Sine ,range is -1.0 +1.0
                    Roll_Sine = float(strItems_ofRow[i]) ;
                    list_ItemsOfRow.append(Roll_Sine);    
                 
                elif i==11 : # Yaw_Cosine ,range is -1.0 +1.0
                    Yaw_Cosine = float(strItems_ofRow[i]) ;
                    list_ItemsOfRow.append(Yaw_Cosine);
                    
                elif i==12 : # Yaw_Sine ,range is -1.0 +1.0
                    Yaw_Sine = float(strItems_ofRow[i]) ;
                    list_ItemsOfRow.append(Yaw_Sine);    

                elif i==13 : # RotationX_Cosine ,range is -1.0 +1.0
                    RotationX_Cosine = float(strItems_ofRow[i]) ;
                    list_ItemsOfRow.append(RotationX_Cosine);


                elif i==14 : # RotationX_Sine ,range is -1.0 +1.0
                    RotationX_Sine = float(strItems_ofRow[i]) ;
                    list_ItemsOfRow.append(RotationX_Sine); 


                elif i==15 : # RotationY_Cosine ,range is -1.0 +1.0
                    RotationY_Cosine = float(strItems_ofRow[i]) ;
                    list_ItemsOfRow.append(RotationY_Cosine);


                elif i==16 : # RotationY_Sine ,range is -1.0 +1.0
                    RotationY_Sine = float(strItems_ofRow[i]) ;
                    list_ItemsOfRow.append(RotationY_Sine);  


                elif i==17 : # RotationZ_Cosine ,range is -1.0 +1.0
                    RotationZ_Cosine = float(strItems_ofRow[i]) ;
                    list_ItemsOfRow.append(RotationZ_Cosine);


                elif i==18 : # RotationZ_Sine ,range is -1.0 +1.0
                    RotationZ_Sine = float(strItems_ofRow[i]) ;
                    list_ItemsOfRow.append(RotationZ_Sine);  

                elif i==19 : # AccelerationX ,range is -1.0 +1.0
                    AccelerationX = float(strItems_ofRow[i]) ;
                    list_ItemsOfRow.append(AccelerationX); 


                elif i==20 : # AccelerationY ,range is -1.0 +1.0
                    AccelerationY = float(strItems_ofRow[i]) ;
                    list_ItemsOfRow.append(AccelerationY); 


                elif i==21 : # AccelerationZ ,range is -1.0 +1.0
                    AccelerationZ= float(strItems_ofRow[i]) ;
                    list_ItemsOfRow.append(AccelerationZ);
                    
                elif i==22 : # GarvityX ,range is -1.0 +1.0
                    GarvityX= float(strItems_ofRow[i]) ;
                    list_ItemsOfRow.append(GarvityX);


                elif i==23 : # GarvityY ,range is -1.0 +1.0
                    GarvityY= float(strItems_ofRow[i]) ;
                    list_ItemsOfRow.append(GarvityY);


                elif i==24 : # GarvityZ ,range is -1.0 +1.0
                    GarvityZ= float(strItems_ofRow[i]) ;
                    list_ItemsOfRow.append(GarvityZ);    
                    
                #####################################################3
                # label columns : Cos & Sin
                
                elif i==26 : # label Cos    
                    TargetAngle_Cosine = float(strItems_ofRow[i]) ;
                    list_ItemsOfRow.append ( TargetAngle_Cosine )
                elif i==27 : # label Sin
                    TargetAngle_Sine = float(strItems_ofRow[i]) ;
                    list_ItemsOfRow.append ( TargetAngle_Sine )
                
            
            list_NormalizedRowsOfFile.append(list_ItemsOfRow)
            # print ('end of list_NormalizedRowsOfFile \n');
                    
    f.close();
    return list_NormalizedRowsOfFile;

def collectAllCSVFiles_to_Dict ( strDir_Training , dict_strAngleRadius_listRowsOfFile ):
    subAll_of_strDir = list( os.walk( top = strDir_Training ) ) ;
    strlist_AllFileNames = subAll_of_strDir[0][2];
        
    for filename in strlist_AllFileNames :
        #TargetAngleOfNorthDirection_deg ,Radius_m  , _ = re.split('-|.txt_output.csv' ,filename);
        TargetAngleOfNorthDirection_deg_and_Radius_m  , _ = re.split('.txt_output.csv' ,filename);
        TargetAngleOfNorthDirection_deg_and_Radius_m += ' file'
        
        strFullFilePath = os.path.join(strDir_Training , filename) ;
        list_NormalizedRowsOfOneFile = processFile_To_NormalizedList(strFullFilePath);
        dict_strAngleRadius_listRowsOfFile [TargetAngleOfNorthDirection_deg_and_Radius_m] = list_NormalizedRowsOfOneFile ;
        # print ('end of one file loop \n');
        

def GenerateClip_bySpecificLength_fromRawDict (dict_strAngleRadius_listRowsOfFile , 
                                               list_TrainingDataBy_ClipLength ,
                                               ClipLength ,
                                               InputItemsLength ) :
    
    for list_SensingRow in dict_strAngleRadius_listRowsOfFile.values():
        EndIndex = len(list_SensingRow) - ClipLength + 1  ;
    
        for i in range(0 ,EndIndex ):
            list_OneClip = [] ;
            LabelCosSine_of_OneClip = [] ;
            i2_EndIndex = i + ClipLength -1 ;
            
            for i2 in range(i ,i2_EndIndex+1):
                row = list_SensingRow[i2][ : InputItemsLength ] ;
                list_OneClip.append(row) ;
                if i2 == i2_EndIndex :
                    label_Row = list_SensingRow[i2][ InputItemsLength :  ]   
                    LabelCosSine_of_OneClip =  label_Row  ;
                    
            list_OneClip = numpy.array ( list_OneClip  ).astype(numpy.float32) ;
            list_OneClip = list_OneClip.reshape(ClipLength ,InputItemsLength ,1 );
            LabelCosSine_of_OneClip = numpy.array ( LabelCosSine_of_OneClip  ).astype(numpy.float32) ;
            
            tupleTmp = tuple( (list_OneClip ,LabelCosSine_of_OneClip) );
            list_TrainingDataBy_ClipLength.append( tupleTmp );
            
        # print ('end of  range(0 ,EndIndex ) loop \n');           
        
    # print ('end of dict_strAngleRadius_listRowsOfFile.values() loop \n');               


def separate_Clip_and_Label ( listTuple_TrainingDataBy_ClipLength ):
    train_Clips = [] ;
    train_Labels = [] ;
    
    for tup in listTuple_TrainingDataBy_ClipLength:
        train_Clips.append( tup[0] )
        train_Labels.append( tup[1] )
        
    train_Clips = numpy.array ( train_Clips  ).astype(numpy.float32) ;    
    train_Labels = numpy.array ( train_Labels  ).astype(numpy.float32) ;   
    
    return  train_Clips, train_Labels ;
    
    
        
#################################################################################################

def readCSV_trainingSeqDataSet_byLabeling_CosSinFloatNum( strDir_Training \
                                                         ,ClipLength = 16 \
                                                         ,InputItemsLength = 6 \
                                                        ): 

    print ('start of read CSVs: \n')
    if strDir_Training == '' :
        #strDir_Training = '2018.03.10_interpolation by time/_processed half 180 compass csv/'  ;
        strDir_Training = '2018.03.10 (Cosine, Sine)/2018.03.10_加了TargetDirection/'
    
    dict_strAngleRadius_listRowsOfFile = dict();
    collectAllCSVFiles_to_Dict( strDir_Training , dict_strAngleRadius_listRowsOfFile )
    
    DataSet.ClipLength = ClipLength ;
    DataSet.InputItemsLength = InputItemsLength ;
    
    
    
    GenerateClip_bySpecificLength_fromRawDict (dict_strAngleRadius_listRowsOfFile , 
                                               DataSet.m_list_TrainingDataBy_ClipLength ,
                                               ClipLength ,
                                               InputItemsLength ) ;
    
    train_Clips , train_Labels = separate_Clip_and_Label( DataSet.m_list_TrainingDataBy_ClipLength );
    print( ' => All  samples count: %d \n' % len(train_Clips) );
    
    train_DataSet = DataSet(train_Clips ,train_Labels ,dtype=dtypes.float32 ,reshape=False)                                               
    
    print ('end of read CSVs');    
    return train_DataSet ;
        
   


