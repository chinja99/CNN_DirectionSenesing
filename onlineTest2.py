# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 22:57:04 2018

@author: chinja
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import math

#from tensorflow.examples.tutorials.mnist import input_data
#import numpy 
#import InputCSVfor0310
import InputCSV
import Cnn2D


#from tensorflow.python.framework import graph_util

import tensorflow as tf
##import matplotlib.image as matimg


def main(_):

  InputItemsLength = 24
  ClipLength = 16
  SZIE_W = InputItemsLength
  CHANNEL_NUM = 1
  CNN_0Dim_SIZE = None
  numOutputVars  =  2; #Cos Sin
  
  
  # Import data
  print("start of read data")
  Dataset_test0 = \
              InputCSV.readCSV_trainingSeqDataSet_byLabeling_CosSinFloatNum (
                      strDir_Training = 'test/test0/'    \
                      , ClipLength = ClipLength  \
                      , InputItemsLength = InputItemsLength );
                      
  test_batchClips, test_batchLabels = Dataset_test0.next_batch( len(Dataset_test0.clips) )    
  
  
  ## check point files path:
  strRoot_Models = 'models/'     
  strNewVersionPath_FoldName = 'chkpt v0 from data v2018-0310/'
  strChkptFileName = 'model_DirectionSensing.ckpt'
  
  strChkptFolderPath = strRoot_Models + strNewVersionPath_FoldName + strChkptFileName ;  
 
  # Create the model
  with tf.Graph().as_default():
  #with graph.as_default():
  
    clip = tf.placeholder(tf.float32, [CNN_0Dim_SIZE, ClipLength, SZIE_W ,CHANNEL_NUM], name='placeHolder_x')
    y_label = tf.placeholder(tf.float32, [CNN_0Dim_SIZE, numOutputVars ], name='placeHolder_y')  
    
    conv_result = Cnn2D.SeqCNN2D (clip, numOutputVars, 1.0) 
    y_conv_result = tf.identity(conv_result, name='y_conv_output_')
   
 
   
    print(clip)
    print(y_label)
    print(y_conv_result)
 
    with tf.name_scope('loss'):
        avg2NodesLoss = tf.reduce_mean(  tf.sqrt( tf.squared_difference( y_conv_result , y_label) )   )
   # with tf.name_scope('angle_loss'):       
    #    angle_loss = tf.sqrt(  tf.squared_difference( angleConv , angleLabel)  )
        #tf.summary.scalar('entropy_loss', loss)
        
    #with tf.name_scope('accuracy'):
    #    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_conv_result, y_label), tf.float32))
  
   # learning_rate = 1e-4
   # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    
    saver = tf.train.Saver()
   
    #config.gpu_options.allow_growth = True
    with tf.Session(config= tf.ConfigProto() ) as sess:    
 
        saver.restore(sess, strChkptFolderPath)        
       
        testConv, avg2NodesLoss_out = sess.run( [y_conv_result , avg2NodesLoss], feed_dict={clip : test_batchClips, y_label :test_batchLabels});
        

        average_DegreeLoss=0.0
        
        for j in range ( len(testConv)):
                
            if j==64 : # only for '0-test.txt_output.csv' 需要特別口頭跟羣哲說明這段的用意
                print('j==64 ----------------------------------------------');         
                
            testDegree  = tf.multiply( tf.angle( tf.complex( testConv[j,0] ,testConv[j,1]))  ,
                                       180.0/math.pi ) ;
                                                            
            tmpTestDeg_out = sess.run([testDegree])[0] 
            print( 'test    out deg : %.8f' % (tmpTestDeg_out ) );
            #-------------------------------------
            #testDegreeV2 = tf.multiply( tf.atan2( testConv[j,1] ,testConv[j,0])  ,
            #                           180.0/math.pi ) ;
                                                 
            #tmpTestDegV2_out = sess.run([testDegreeV2])[0]                                                  
            #print( 'testV2  out deg : %.8f' % (tmpTestDegV2_out ) );
            
            #***********************************************************
            LabelDegree = tf.multiply( tf.angle( tf.complex( test_batchLabels[j,0] ,test_batchLabels[j,1]))  ,
                                       180.0/math.pi ) ;
            
            tmpLabelDegree_out = sess.run([LabelDegree])[0] 
            print( 'Label   out deg : %.8f' % (tmpLabelDegree_out ) );
            #-------------------------------------                                                                
            #LabelDegreeV2 = tf.multiply( tf.atan2( test_batchLabels[j,1] ,test_batchLabels[j,0])  ,
            #                           180.0/math.pi ) ;
            
            #tmpLabelDegreeV2_out = sess.run([LabelDegreeV2])[0] 
            #print( 'Labelv2 out deg : %.8f' % (tmpLabelDegreeV2_out ) );
            
            ################################################################3
            
            angle_Loss = tf.sqrt(  tf.squared_difference( testDegree , LabelDegree)  )
            angle_AbsLoss = tf.abs(  tf.subtract( testDegree , LabelDegree)  )
            
            tmpAngle_Loss_out = sess.run([angle_Loss])[0]
            #tmpAngle_Loss_out1 = sess.run([angle_Loss])
            if tmpAngle_Loss_out > 180.0 :
                # 例外狀況修正：
                angle_Loss = 360.0 - angle_Loss;
                angle_AbsLoss = 360.0 - angle_AbsLoss;
                tmpAngle_Loss_out = sess.run( [ angle_Loss ])[0] ;
                
            
            print('j=%d, angle Loss  (degs)  is: %.9f ' % (j,  tmpAngle_Loss_out )  )
            #angle_AbsLoss_out = sess.run([angle_AbsLoss ]) ;
            #print('j=%d, angle AbsLoss is: %.9f' % (j,  angle_AbsLoss_out )  )
            
            average_DegreeLoss += tmpAngle_Loss_out                 
                
        print('Average 2Node Loss: %.5f' % ( avg2NodesLoss_out))
        print('Average Angle(degs) Loss is: %.5f ' % ( average_DegreeLoss/len(testConv)  ))
        
               
         
        print('end of testing')
        
     
        
        
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
        
        
        
        
        
        
        
        
  