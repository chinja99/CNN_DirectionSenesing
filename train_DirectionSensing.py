# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys


#from tensorflow.examples.tutorials.mnist import input_data
import numpy 
import InputCSV
import Cnn2D


from tensorflow.python.framework import graph_util

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
                
  print("test data - end")                      
  
  InSeqClips_Dataset_train = \
              InputCSV.readCSV_trainingSeqDataSet_byLabeling_CosSinFloatNum (
                      strDir_Training = '2018.03.10 (Cosine, Sine)/2018.03.10_加了TargetDirection/'    \
                      , ClipLength = ClipLength  \
                      , InputItemsLength = InputItemsLength );
  
                      
  
  # test_dir="/tmp/code_3DCNN/Data_3DCNN/Test/",
  print("end of read data")

  
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
    
    conv_result = Cnn2D.SeqCNN2D (clip, numOutputVars, 0.5) 
    y_conv_result = tf.identity(conv_result, name='y_conv_output_')
    
    print(clip)
    print(y_label)
    print(y_conv_result)
    
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(   tf.squared_difference( y_conv_result , y_label)   )
        #tf.summary.scalar('entropy_loss', loss)
        
    #with tf.name_scope('accuracy'):
    #    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_conv_result, y_label), tf.float32))
  
    learning_rate = 1e-4
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    
    saver = tf.train.Saver()
    
    #config.gpu_options.allow_growth = True
    with tf.Session(config= tf.ConfigProto() ) as sess:
        
        ##################################################################################################
        saver.restore(sess, strChkptFolderPath)
        ###    使用  saver.restore，下面行就要註解起來，反之第一次執行，則要使用下面行，註解上面的saver.restore
        #sess.run(tf.global_variables_initializer()) ; sess.run(tf.local_variables_initializer()) ;
        ##################################################################################################
        
        step = 0;
        BATCH_SIZE = 50;
        End_of_EPOCH_NUM = 500000
        
        SaveModelTiming_MultipleNumberOfEpoch = 50;
        
        for epoch in range(End_of_EPOCH_NUM):
            
            loss_epoch = 0
            
            for i in range( len(InSeqClips_Dataset_train.clips) // BATCH_SIZE):
                step += 1
                
                batchClips, batchLabels = InSeqClips_Dataset_train.next_batch( BATCH_SIZE )

                _, loss_out = sess.run([optimizer, loss], feed_dict={clip : batchClips, y_label : batchLabels});
                
                loss_epoch += loss_out
                
                if i % 1 == 0:
                    #print('Epoch %d, Batch %d: Loss is %.9f; Accuracy is %.3f'%(epoch, i, loss_out, accuracy_out))
                    print('Epoch %d, Batch %d: Loss is %.5f; '%(epoch, i, loss_out))
                    
                ############ for batch i loop ######################################################################
            print('\n Epoch %d: Average loss is: %.9f;  \n'%(epoch, loss_epoch / (len(InSeqClips_Dataset_train.clips) // BATCH_SIZE) ))        
           
          
          
            # Save checkpoint, graph.pb 
            if epoch % SaveModelTiming_MultipleNumberOfEpoch == 0 and epoch != 0 :
                print("\n Save checkpoint ...\n")
                #saver = tf.train.Saver()
                saver.save(sess, strChkptFolderPath) ;
                output_node_names = 'y_conv_output_'
                #print(output_node_names) ;
                graph = tf.get_default_graph() ;
                input_graph_def = graph.as_graph_def() ;
                output_graph_def = graph_util.convert_variables_to_constants(
                    sess, # The session is used to retrieve the weights
                    input_graph_def, # The graph_def is used to retrieve the nodes 
                    output_node_names.split(",") # The output node names are used to select the usefull nodes
                    ); 
                        
                # save PB file   ----------------------------------------------------
                output_graph = "models/freeze/frozen_model_DirectionSensing.pb"
                # Finally we serialize and dump the output graph to the filesystem
                with tf.gfile.GFile(output_graph, "wb") as f:
                    f.write( output_graph_def.SerializeToString() )
                print("%d ops in the final graph." % len(output_graph_def.node))        
            # end of Save checkpoint, graph.pb     
            
            ## Shuffel_and_Reassign_Dataset       
            if  epoch % 2 == 0 and epoch != 0 :
                InSeqClips_Dataset_train = InputCSV.DataSet.Shuffel_and_Reassign_Dataset();        
                
        ########## end of for epoch loop ######################################################################################
        print("start for save")
          
        # Save checkpoint, graph.pb and tensorboard
        #saver = tf.train.Saver()
        saver.save(sess, strChkptFolderPath) 
        output_node_names = 'y_conv_output_'
        print(output_node_names)
        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            input_graph_def, # The graph_def is used to retrieve the nodes 
            output_node_names.split(",") # The output node names are used to select the usefull nodes
            ) 
        
        output_graph = "models/freeze/frozen_model_DirectionSensing.pb"
        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph." % len(output_graph_def.node))
        
      
          
        tf.train.write_graph(sess.graph.as_graph_def(), "models/freeze/", "frozen_model_DirectionSensing.pb" ,as_text=False)  
        print("end of the save")
    ####################################################################

    
  
  
  print("---The end of our testing---")
    
    

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
