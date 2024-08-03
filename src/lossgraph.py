import os
import re

import numpy as np
from pylab import *

from configs import *


def readfile(filename):
    print(os.getcwd())

    print(filename)
    test_file = open(filename, 'r+', encoding='utf-8')
    test = test_file.read()
    step = re.findall(r'step (\d+), va', test)
    loss = re.findall(r'validation loss = (.{5})', test)
    step =[int(i)/1000 for i in step]
    loss =[20 * math.log10(255.0 / math.sqrt(float(i))) for i in loss]

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
    X1, Y1 = readfile('./Image2X/mod2plusY/loss.log')
    X2, Y2 = readfile('./Image2X/mod2plus/loss.log')
    X3, Y3 = readfile('./Image2X/modplus/loss.log')
    #figure(figsize=(800, 500), dpi=72)
    subplot(1, 1, 1)
    ylim(30,80)
    xticks(np.linspace(0,100,11))
    yticks(np.linspace(30,35,21))
    plot(X1, Y1, color='r')
    plot(X2, Y2, color='g')
    plot(X3, Y3, color='b')
    #scatter(X2,Y2,color='b')
    savefig("exercice_2.png", dpi=72)
    show()


if __name__ == '__main__':
    main()
