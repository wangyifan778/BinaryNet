import scipy.io
import fileinput
import glob
import numpy as np

with open('./binaryNet_int_val.txt', 'w') as out:
    matpath = './val.mat'
    resnet_l = scipy.io.loadmat(matpath)

  ##########################bn1#########################
    y = resnet_l["bn1"][0][0][0]
    for i in range(128):
      out.write(str(y[i][0]) + ' ')
    out.write('\n')
    ##########################bn2#########################
    y = resnet_l["bn2"][0][0][0]
    for i in range(128):
      out.write(str(y[i][0]) + ' ')
    out.write('\n')
    ##########################bn3#########################
    y = resnet_l["bn3"][0][0][0]
    for i in range(256):
      out.write(str(y[i][0]) + ' ')
    out.write('\n')
    ##########################bn4#########################
    y = resnet_l["bn4"][0][0][0]
    for i in range(256):
      out.write(str(y[i][0]) + ' ')
    out.write('\n')
    ##########################bn5#########################
    y = resnet_l["bn5"][0][0][0]
    for i in range(512):
      out.write(str(y[i][0]) + ' ')
    out.write('\n')
    ##########################bn6#########################
    y = resnet_l["bn6"][0][0][0]
    for i in range(512):
      out.write(str(y[i][0]) + ' ')
    out.write('\n')
    ##########################bn7#########################
    y = resnet_l["bn7"][0][0][0]
    for i in range(1024):
      out.write(str(y[i][0]) + ' ')
    out.write('\n')
    ##########################bn8#########################
    y = resnet_l["bn8"][0][0][0]
    for i in range(10):
      out.write(str(y[i][0]) + ' ')
    out.write('\n')
    y = resnet_l["bn8"][0][0][1]
    for i in range(10):
      out.write(str(y[i][0]) + ' ')
    out.write('\n')
