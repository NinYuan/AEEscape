#coding:UTF-8
import pickle,os
from scipy import *
import math
import pickle
import seaborn
import numpy as np
import matplotlib
from scipy.stats import pearsonr
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def compareTwoDataset(ExperimentV,Predictv,filepath,xlable,ylable,figsize):
    # print name
    # print value
    plt.clf()
    fig, ax = plt.subplots(figsize=(figsize / 2.54, figsize / 2.54))
    ax.scatter(ExperimentV,Predictv)
    print('compare')
    print(np.max(ExperimentV))
    print(np.max(Predictv))
    print(np.min(ExperimentV))
    print(np.min(Predictv))
    #prcc=pearsonr(name,value)
    #print(prcc)
    #prcc=round(pearsonr(ExperimentV,Predictv)[0],3)
    nas = np.logical_or(np.isinf(ExperimentV), np.isinf(Predictv))
    # prcc=round(pearsonr(ExperimentV,Predictv)[0],3)
    prcc = round(pearsonr(ExperimentV[~nas], Predictv[~nas])[0], 3)
    #print(name)
    #print(value)
    print('prcc')
    print(prcc)
    #plt.title(tilename+str(prcc))
    # plt.ylabel(ylable)
    # plt.xlabel(xlable)
    plt.xlabel(xlable)
    plt.xticks()
    plt.ylabel(ylable)
    plt.yticks()
    s=r'$\rho$'+' = '+str(prcc)
    ExperimentV[np.isinf(ExperimentV)]=0
    Predictv[np.isinf(Predictv)] = 0
    maxExp=max(ExperimentV)
    maxPre = max(Predictv)
    textlocx=maxExp/2
    textlocy = maxPre
    ax.text(textlocx,textlocy,s)
    plt.tight_layout()
    plt.savefig(filepath,dpi=400)
    #plt.savefig('Wrky1N420180322_log2.pdf')
    return prcc



def printPicE(fMatrix,outpath,labeltxt,vmax,vmin,figsize):

    plt.clf()
    #nparrayf=np.array(fMatrix)
    #print(outpath)
    #print(fMatrix.shape)
    #print(fMatrix.type)
    Nsize=int(math.log2(fMatrix.shape[0]))
    #print(Nsize)
    lable1=int(math.sqrt(4**Nsize))-1

    fig, ax = plt.subplots(figsize=(figsize / 2.54*1.2, figsize / 2.54))

    #data=ax.imshow(fMatrix)
    #print(np.max(fMatrix))
    #print(np.min(fMatrix))
    #data = ax.imshow(fMatrix,vmax=vmax,vmin=vmin)
    #seaborn.heatmap(fMatrix, ax=ax, cmap="rainbow")
    seaborn.heatmap(fMatrix,  ax=ax,cmap="viridis_r")

    #seaborn.heatmap(fMatrix,vmax=vmax,vmin=vmin,ax=ax,cmap="rainbow")

    plt.xticks([])
    plt.yticks([])
    #plt.colorbar( format='%.2e',label='Output Div Input signal')
    #cb = fig.colorbar(data,format='%.1f', label=labeltxt)
    #cb=fig.colorbar(fMatrix ,label=labeltxt)
    #cb.ax.tick_params(labelsize=4)
    #cb.set_ticks([0,1])
    colornm='red'

    text_params = {'ha': 'center', 'va': 'center', 'family': 'sans-serif'}
    # plt.text(-1, -1, 'G' * Nsize, color=colornm, **text_params)
    # plt.text(lable1, -1, 'C' * Nsize, color=colornm, **text_params)
    # plt.text(-1, lable1+1, 'A' * Nsize, color=colornm, **text_params)
    # plt.text(lable1 ,lable1+1, 'T' * Nsize, color=colornm, **text_params)
    # plt.text(-0.5, -0.5, 'G' * Nsize, color=colornm, **text_params)
    # plt.text(lable1, -0.5, 'C' * Nsize, color=colornm, **text_params)
    # plt.text(-0.5, lable1 + 0.5, 'A' * Nsize, color=colornm, **text_params)
    # plt.text(lable1, lable1 + 0.5, 'T' * Nsize, color=colornm, **text_params)

    if Nsize<6:
        plt.text(-0 + 0.5, -0 + 0.5, 'G' * Nsize, color=colornm, **text_params)
        plt.text(lable1 + 0.5, -0 + 0.5, 'C' * Nsize, color=colornm, **text_params)
        plt.text(-0 + 0.5, lable1 + 0.5, 'A' * Nsize, color=colornm, **text_params)
        plt.text(lable1 + 0.5, lable1 + 0.5, 'T' * Nsize, color=colornm, **text_params)
    else:
        plt.text(-0, -0, 'G' * Nsize, color=colornm, **text_params)
        plt.text(lable1-1, -0, 'C' * Nsize, color=colornm, **text_params)
        plt.text(-0, lable1 , 'A' * Nsize, color=colornm, **text_params)
        plt.text(lable1-1, lable1 , 'T' * Nsize, color=colornm, **text_params)

    #plt.title(title)

    #plt.xlim((0, 14000))
    plt.tight_layout()
    plt.savefig(outpath,dpi=400)

    #plt.savefig(outpath,dpi=400,figsize=(9 / 2.54, 9 / 2.54))
    plt.close()


def predictCompare(SBInfo,SInfo,Predictpath,outdir,experimentIndex,figuresize):
    ExperimentV = np.array(SBInfo[0])
    PsProbs = np.array(SInfo[0])
    # relativeIntensity=ExperimentV/PsProbs
    relativeIntensity = np.divide(ExperimentV, PsProbs, out=np.zeros_like(ExperimentV), where=PsProbs != 0)  #
    ExperimentV = -np.log2(relativeIntensity.astype('float'))
    Predictv = -np.log(np.loadtxt(Predictpath).flatten())

    filepath = outdir + 'Ecorr' + Predictpath.split('/')[-1][:-4] + '.png'
    xlabel = 'Binding energy from experiments'
    ylabel = 'Predicted binding energy'
    compareTwoDataset(ExperimentV, Predictv, filepath, xlabel, ylabel, figuresize)

    filepath = outdir + 'CHcorr' + Predictpath.split('/')[-1][:-4] + '.png'
    xlabel = 'KaScape实验直接计算的结合能'
    ylabel = '预测的结合能'
    compareTwoDataset(ExperimentV, Predictv, filepath, xlabel, ylabel, figuresize)

    labeltxt = ''
    vmax = ''
    vmin = ''
    outpath = outdir + Predictpath.split('/')[-1][:-4] + 'E.png'
    pfMatrix = -np.log(np.loadtxt(Predictpath))
    printPicE(pfMatrix, outpath, labeltxt, vmax, vmin, figuresize)

    outpath = outdir + experimentIndex + 'E.png'
    seqnum=len(ExperimentV)
    numlen=int(np.sqrt(seqnum))
    #print(seqnum)
    ExperimentVr=np.reshape(ExperimentV,(numlen,numlen))
    #efMatrix = -np.log2(np.loadtxt(Experimentpath))
    printPicE(ExperimentVr, outpath, labeltxt, vmax, vmin, figuresize)