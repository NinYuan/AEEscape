#coding:UTF-8
from scipy import *

import matplotlib
from scipy.stats import pearsonr
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

font={'size':12}
fs=10


def writeProbKmer(subkmer,outfilepath):
    fw = open(outfilepath, 'w')
    for i in range(len(subkmer)):
        for j in range(len(subkmer)):
            fw.write('%.6f'%float(subkmer[i][j]) + "\t")

        fw.write('\n')

def compareTwoDataset(ExperimentV,Predictv,tilename,filepath,xlable,ylable):

    plt.clf()
    plt.scatter(ExperimentV,Predictv)

    prcc=round(pearsonr(ExperimentV,Predictv)[0],3)

    print('prcc')
    print(prcc)
    plt.title(tilename+str(prcc))

    plt.xlabel(xlable, fontdict=font)
    plt.xticks(fontsize=fs)
    plt.ylabel(ylable, fontdict=font)
    plt.yticks(fontsize=fs)
    s=r'$\rho$'+' = '+str(prcc)
    maxExp=max(ExperimentV)
    maxPre = max(Predictv)
    textlocx=maxExp
    textlocy = maxPre
    plt.text(textlocx,textlocy,s,fontsize=15)
    plt.tight_layout()
    plt.savefig(filepath)
    return prcc