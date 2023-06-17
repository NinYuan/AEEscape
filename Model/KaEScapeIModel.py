# coding: utf-8
from scipy import *
import numpy as np

def Gradient(f, x0, h=1e-8):
#def Gradient(f, x0, h=0.1):
    n = len(x0)
    g = zeros(n)
    for i in range(n):
        x = array(x0)
        x[i] -= h
        g[i] -= f(x)
        x = array(x0)
        x[i] += h
        g[i] += f(x)
    g /= 2 * h
    return g



class Layer(object):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n + 2

    @classmethod
    def error(cls, output, target):
        return ((output - target) ** 2).sum() / 2

    def RjthetaOutput(self, w):
        Esj = w[:self.n].T
        mu = w[-2]
        scale = w[-1]
        return exp(-Esj)/(exp(-Esj)+exp(-mu))/scale

    def RjTarget(self, SBInfo,SInfo,w):
        Ejs=w[:-2]
        mu = w[-2]
        PTsjBs = self.getPTsjBs(SBInfo, Ejs.T) #加总不一定为1
        PTsjProb = self.getMotifjProb(SInfo, Ejs.T, mu)
        Rjs=np.divide(PTsjBs, PTsjProb, out=np.zeros_like(PTsjBs), where=PTsjProb != 0)
        return Rjs



    def getPTsjBs(self,SBInfo, Ejs):

        probs=SBInfo[0]
        SeqInfo=array(SBInfo[1])

        EjsRepeat=np.tile(Ejs,(SeqInfo.shape[0],1))

        print(SeqInfo.shape)
        print(len(Ejs))

        #每条序列每个位置每种motif值
        SeqijKaEach=(SeqInfo.reshape(len(SeqInfo),len(Ejs))*exp(-EjsRepeat))

        #计算每条序列的Ka值
        KatotalseqiEach=(SeqijKaEach).sum(axis=1)

        KatotalRepeat=KatotalseqiEach.reshape(len(KatotalseqiEach),1).repeat(len(Ejs),axis=1)

        probsRepeat=probs.reshape(len(probs),1).repeat(len(Ejs),axis=1)
        Pmotifjseqi=SeqijKaEach/KatotalRepeat*probsRepeat
        Pmotifjseq=Pmotifjseqi.sum(axis=0)

        return Pmotifjseq


    def getMotifjProb(self,SInfo, Ejs, chemicalPotential):
        # 所有序列存在motifj的概率
        probs = SInfo[0]
        SeqInfo = array(SInfo[1])

        EjsRepeat = np.tile(Ejs, (SeqInfo.shape[0], 1))

        # 每条序列每个位置每种motif值
        SeqInfoReshape=SeqInfo.reshape(len(SeqInfo), len(Ejs))
        SeqijKaEach = SeqInfoReshape * exp(-EjsRepeat)
        SeqijKaMuEach=(SeqijKaEach+exp(-chemicalPotential))*SeqInfoReshape

        # 计算每条序列的Ka值
        KatotalseqiEach = SeqijKaEach.sum(axis=1)+exp(-chemicalPotential)

        KatotalRepeat = KatotalseqiEach.reshape(len(KatotalseqiEach), 1).repeat(len(Ejs), axis=1)

        wijl=SeqijKaMuEach/KatotalRepeat #256*16

        probsRepeat = repeat(probs.reshape(len(probs), 1), len(Ejs), axis=1)
        seqwij=wijl*probsRepeat
        return seqwij.sum(axis=0)


class TrainingClosure(object):

    def __init__(self, NN, SBInfo,SInfo,cmarray,target):
        self.NN = NN  # model
        """:type: Layer"""
        self.SBInfo = SBInfo  # data
        self.SInfo = SInfo
        self.cmarray = cmarray
        self.target=target

    def lossfunc(self, w):

        O = self.NN.RjthetaOutput(w)
        loss = self.NN.error(O, self.target)
        if isnan(loss):
            loss = Inf
        return loss


    def gradient(self, w):
        return Gradient(self.lossfunc, w)

    def hessian(self, w, h=1e-4):
        N = len(self.NN)
        H = zeros((N, N))
        for i in range(N):
            W = array(w)
            W[i] -= h
            H[:, i] = -self.gradient(W)
            W = array(w)
            W[i] += h
            H[:, i] += self.gradient(W)
        H /= (2 * h)
        H = (H + H.T) / 2
        return H


    def InvCov(self, w,h=1e-6):
        H = self.hessian(w, h)
        valuablenum=len(self.cmarray)*len(self.SBInfo[1][0])+2
        #m = len(self.target.T) - valuablenum #总样本量-模型参数
        m = len(self.SBInfo[1]) - valuablenum  # 总样本量-模型参数
        sigma2 = self.lossfunc(w) / m  # unbiased estimate of sigma^2
        return H / sigma2


