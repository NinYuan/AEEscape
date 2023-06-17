#coding:UTF-8
from scipy import *
from IOTool.OutTool import writeProbKmer,compareTwoDataset
from IOTool.Draw2dfigure import printPic
import numpy as np



def writePredictKaScapeProbability(probs,outfilepath):
    num = int(sqrt(len(probs)))
    kmerdata = probs.reshape(num, num)
    writeProbKmer(kmerdata, outfilepath)

def predictKaScape(Kas,mu,Pbscale,UInfo,SBInfo,outpathfile,outfigurepathfile,title):

    SeqInfo=array(UInfo[1])
    SeqijKaEach = (SeqInfo.reshape(len(SeqInfo), len(Kas)) * Kas)

    # 计算每条序列的Ka值
    Katotals = (SeqijKaEach).sum(axis=1)

    UProbs=array(UInfo[0])
    BProbs=array(SBInfo[0])
    PsProbsDivPb=BProbs+Pbscale*UProbs
    PTSB=(Katotals/(Katotals+exp(-mu)))*PsProbsDivPb
    #形成KaScape概率图
    outpath=outpathfile+'PTSB'+'.txt'
    outfigurepath=outfigurepathfile+'PTSB'+'.pdf'
    writePredictKaScapeProbability(PTSB, outpath)
    printPic(outpath, outfigurepath, title)

    outpathKa = outpathfile + 'Ka' + '.txt'
    outfigurepathKa = outfigurepathfile+ 'Ka' + '.pdf'
    writePredictKaScapeProbability(Katotals, outpathKa)
    printPic(outpathKa, outfigurepathKa, title)

    BU=Katotals*exp(mu)*Pbscale #通过Ka 预测 bound/unbound的值 [T]=exp(mu)

    outpathBU = outpathfile + 'BdivU' + '.txt'
    outfigurepathBU = outfigurepathfile + 'BdivU' + '.pdf'
    writePredictKaScapeProbability(BU, outpathBU)
    printPic(outpathBU, outfigurepathBU, title)

    return PTSB,BU

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
    xlabel='Experiment unbound div bound'
    ylabel='Predict unbound div bound'
    prcc=compareTwoDataset(relativeIntensity, Ka, tilename, filepath, xlabel, ylabel)
    return prcc
