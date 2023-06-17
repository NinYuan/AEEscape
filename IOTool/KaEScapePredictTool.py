#coding:UTF-8
from scipy import *
from IOTool.OutTool import writeProbKmer,compareTwoDataset
from IOTool.Draw2dfigure import printPic
import numpy as np



def writePredictKaScapeProbability(probs,outfilepath):
    num = int(sqrt(len(probs)))
    #print(num)
    kmerdata = probs.reshape(num, num)
    #print(kmerdata)
    writeProbKmer(kmerdata, outfilepath)

def predictKaScape(Kas,mu,Pb,SInfo,outpathfile,outfigurepathfile,title):
    SeqInfo=array(SInfo[1])
    SeqijKaEach = (SeqInfo.reshape(len(SeqInfo), len(Kas)) * Kas)

    # 计算每条序列的Ka值
    Katotals = (SeqijKaEach).sum(axis=1)

    PsProbs=array(SInfo[0])
    PTSB=(Katotals/(Katotals+exp(-mu)))*PsProbs/Pb
    #print(PTSB)
    #形成KaScape概率图
    outpath=outpathfile+'PTSB'+'.txt'
    outfigurepath=outfigurepathfile+'PTSB'+'.pdf'
    writePredictKaScapeProbability(PTSB, outpath)
    printPic(outpath, outfigurepath, title)

    outpathKa = outpathfile + 'Ka' + '.txt'
    outfigurepathKa = outfigurepathfile+ 'Ka' + '.pdf'
    writePredictKaScapeProbability(Katotals, outpathKa)
    printPic(outpathKa, outfigurepathKa, title)

    RI=(Katotals/(Katotals+exp(-mu)))/Pb

    outpathRI = outpathfile + 'RI' + '.txt'
    outfigurepathRI = outfigurepathfile + 'RI' + '.pdf'
    writePredictKaScapeProbability(RI, outpathRI)
    printPic(outpathRI, outfigurepathRI, title)

    return PTSB,RI

def compareKaScape(PTSB,SBInfo,tilename,filepath):
    ExperimentV=SBInfo[0]
    xlable='Experiment Probability P(Si|B)'
    ylable='Predict Probability P(Si|B)'
    prcc=compareTwoDataset(ExperimentV, PTSB, tilename, filepath, xlable, ylable)
    return prcc


def compareKaConstantScape(Ka,SInfo,SBInfo,tilename,filepath):
    ExperimentV=array(SBInfo[0])
    PsProbs = array(SInfo[0])
    relativeIntensity = np.divide(ExperimentV, PsProbs, out=np.zeros_like(ExperimentV), where=PsProbs != 0)  #
    xlabel='Experiment relative intensity'
    ylabel='Predict relative association constant'
    prcc=compareTwoDataset(relativeIntensity, Ka, tilename, filepath, xlabel, ylabel)
    return prcc
