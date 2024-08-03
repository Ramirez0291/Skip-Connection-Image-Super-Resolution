import os
import re

import numpy as np
from pylab import *

from configs import *


def readfile(filename):
    print(os.getcwd())


    test_file = open(filename, 'r+', encoding='utf-8')
    test = test_file.read()
    step = re.findall(r'(\d+). PSNR:', test)
    loss = re.findall(r'. PSNR:(.{6})', test)
    step =[int(i)/10000 for i in step]
    loss =[float(i) for i in loss]

    X = array(step)
    Y = array(loss)
    return X, Y


def main():
    '''
    plot(C, D, color="blue", linewidth=1.0, linestyle="-")
    ylim(-0.1,1.0)
    yticks(np.linspace(-0.1,1.0,12,endpoint=True))
    xlim(0.0,200000.0)
    xticks(np.linspace(0,200000,200001,endpoint=True))
    '''
    X1, Y1 = readfile('./Image2X/mod2plusY/psnr.log')
    X2, Y2 = readfile('./Image2X/mod2plus/psnr.log')
    X3, Y3 = readfile('./Image2X/modplus/psnr.log')
    #X2,Y2=readfile('loss_rescnn.txt')
    #figure(figsize=(800, 500), dpi=72)
    subplot(1, 1, 1)
    ylim(30,35)
    xticks(np.linspace(0,100,11))
    yticks(np.linspace(30,35,21))
    plot(X1, Y1, color='r')
    plot(X2, Y2, color='g')
    plot(X3, Y3, color='b')
    #scatter(X2,Y2,color='b')
    savefig("%s.png"%MODEL_NAME, dpi=72)
    show()


if __name__ == '__main__':
    main()
