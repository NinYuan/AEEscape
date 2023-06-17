#coding:UTF-8
import math
import numpy as np
import matplotlib.pyplot as plt
from IOTool.ReadTool import readData

def printPic(filepath,outpath,title):
    plt.clf()

    fMatrix1 = readData(filepath)

    nparrayf=np.array(fMatrix1)
    Nsize=int(math.log2(nparrayf.shape[0]))

    lable1=int(math.sqrt(4**Nsize))-1

    plt.imshow(fMatrix1)

    plt.xticks([])
    plt.yticks([])
    plt.colorbar( format='%.2e',label='Association constant Ka (M)')
    colornm='r'

    text_params = {'ha': 'center', 'va': 'center', 'family': 'sans-serif',
                   'fontweight': 'bold'}
    plt.text(0, 0, 'G' * Nsize, color=colornm, **text_params)
    plt.text(lable1, 0, 'C' * Nsize, color=colornm, **text_params)
    plt.text(0, lable1, 'A' * Nsize, color=colornm, **text_params)
    plt.text(lable1 ,lable1, 'T' * Nsize, color=colornm, **text_params)
    plt.title(title)

    plt.savefig(outpath)
    plt.close()

