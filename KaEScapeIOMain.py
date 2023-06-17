#coding:UTF-8
import pickle
from scipy import *
from scipy import linalg
import os
from Model.KaEScapeIModel import TrainingClosure,Layer
import numpy as np
from IOTool.ReadTool import readData
from IOTool.Draw2dfigure import printPic
from IOTool.KaEScapePredictTool import compareKaScape, predictKaScape,compareKaConstantScape
import pandas as pd
from multiprocessing import Pool
import matplotlib

matplotlib.use('Agg')


PAIRS = {'A': 'T',
         'C': 'G',
         'G': 'C',
         'T': 'A',
         'N': 'N'}


class DataReader:
    def __init__(self,input_path,flank,length,cmpLath,cmpNpath,Outpath):
        self.input_path=input_path
        self.flank=flank
        self.length=length
        self.cmLpath=cmpLath # k 短序列
        self.cmpNpath = cmpNpath # L 长序列
        self.cmArray=[]
        self.Outpath=Outpath
    def readSeq(self,prob,seqN):
        #seq = self.flank[0] + seqN + self.flank[1]
        seq = seqN
        submotifs = self.getSubMotif(seq)
        return prob,submotifs.tolist()
    #读取数据,按cmArray来排列每一条序列的子序列
    def read_out(self):
        contentmat=np.array(readData(self.input_path)).flatten()
        cMatrix = pickle.load(open(self.cmLpath, "rb"))
        self.cmArray = np.array(cMatrix).flatten()

        cNMatrix = np.array(pickle.load(open(self.cmpNpath, "rb"))).flatten()

        readSeqFunc=np.frompyfunc(self.readSeq, 2, 2)
        probs,subseqs=readSeqFunc(contentmat,cNMatrix)

        pickle.dump([probs, subseqs.tolist()], open(self.Outpath, 'wb'))
        return [probs,subseqs.tolist()]

    def getReverseComplement(self,seq):
        return ''.join(PAIRS[i] for i in reversed(seq))

    def get_subseqindex(self,seq):

        fseq = np.frompyfunc(lambda x: x == seq, 1, 1)
        ret = fseq(self.cmArray)
        return ret

    def getsubseq(self,seq):
        iterlen = len(seq) - self.length + 1
        subseqs = []
        for i in range(iterlen):
            subseq = seq[i:i + self.length]
            subseqs.append(subseq)
        fall = np.frompyfunc(self.get_subseqindex, 1, 1)
        subseqIndex = fall(subseqs)

        return subseqIndex

    def getSubMotif(self,seq):
        # 获取反向互补序列，截取motif,返回motif 列表
        #rc_seq = self.getReverseComplement(seq)
        subseqs = self.getsubseq(seq)
        #rcsubseqs = self.getsubseq(rc_seq)
        #totalseq = np.array([subseqs ,rcsubseqs]).sum(axis=0)
        return subseqs
        #return totalseq

#初始化KaScape motif
#获取KaScape序列，给每个序列初始化一个数
def initKaScape(cmpath,length,TFM,DNAM,PBscale,LseqLen):
    cMatrix = pickle.load(open(cmpath, "rb"))
    cmArray=np.array(cMatrix).flatten()
    totalseqNum=4**length
    paverage=1.0/totalseqNum
    totalweightNum=totalseqNum*LseqLen

    #假设每条DNA序列与一个转录因子结合
    mu = log(TFM-DNAM * PBscale)
    Eave=-log(exp(-mu)*paverage/(1-paverage)) #同一条序列有两条反向互补链
    initvalue=list(np.ones(totalweightNum)*Eave)
    initvalue.extend([mu,PBscale])

    return cmArray,array(initvalue)




def Train(TC,initW,upperconstraint,lowerconstraint,maxIter=5,verbosity=0,solver='scipy_lbfgsb'):
    p = NLP(TC.lossfunc, initW, iprint=verbosity, maxIter=maxIter)
    p.df = TC.gradient
    p.lb = lowerconstraint
    p.ub = upperconstraint
    r = p.solve(solver)
    print (r.ff, r.xf)
    return r.ff,r.xf


def writeKmer(subkmer,outfilepath):
    fw = open(outfilepath, 'w')
    for i in range(len(subkmer)):
        for j in range(len(subkmer)):
            fw.write('%.2f'%exp(-subkmer[i][j]) + "\t")
        fw.write('\n')
def getCIandWrite(outdir,TC,w,motiflen,i,rff,maxIter,experimentIndex):
    # 计算可信度区间

    wCIvalue = TC.InvCov(w)
    try:
        COV = linalg.inv(wCIvalue)
    except linalg.LinAlgError:
        return
    cond = np.diag(COV) < 0

    var = np.where(cond, np.inf, np.diag(COV))
    std = np.sqrt(var)

    muCI = std[-2]
    pbCI = std[-1]
    CIfilenm=outdir+experimentIndex+'sampleCIMotiflen%d' % motiflen + 'loop%d' % i + 'maxIter%d' % maxIter + 'mu%.2e' % muCI + 'pb%.2e' % pbCI + 'error%.2e' % rff

    energyCI = std[:-2]
    outfile=CIfilenm+'KaCI'
    saveEnergy(energyCI, motiflen, outfile,'KaCI'+'e%.2f'%rff)




def saveEnergy(energy,motiflen,outfile,title):
    lenEnergy=int(len(energy)/(4**motiflen))
    for i in range(lenEnergy):
        outfilepath=outfile+'loc'+str(i)+'.txt'
        energyloc=energy[4**motiflen*i:4**motiflen*(i+1)]
        kmerdata = energyloc.reshape(2**motiflen, 2**motiflen)
        writeKmer(kmerdata, outfilepath)
        outfigurepath=outfile+'loc'+str(i)+'.pdf'
        titlepng=title+'loc'+str(i)
        printPic(outfilepath, outfigurepath, titlepng)
    return

def saveResult(w,experimentIndex,motiflen,i,maxIter,rff,outdir,TC,SInfo,SBInfo,date,randomN):
    print('saveResult')
    mu = w[-2]
    pb = w[-1]
    filenm = experimentIndex + 'motiflen%d' % motiflen + 'loop%d' % i + 'maxIter%d' % maxIter + 'mu%.2f' % mu + 'pb%.2f' % pb + 'error%.2e' % rff
    outfilepath = outdir +filenm
    energy = w[:-2]
    outfile=outfilepath+'KaPredict'
    saveEnergy(energy, motiflen, outfile,'KaCI'+'e%.2f'%rff)

    getCIandWrite(outdir, TC, w, motiflen, i, rff, maxIter, experimentIndex)


    pdirname = date + '_p' + str(randomN) + 'by' + str(motiflen)
    outKmerpath = outdir + pdirname + 'PredictProb' + filenm
    title = pdirname + filenm
    kmer2dfigurepath = outdir + pdirname + 'PredictProbFig' + filenm
    PTSB, RI = predictKaScape(exp(-energy), mu, pb, SInfo, outKmerpath, kmer2dfigurepath, title)

    tilename = pdirname + filenm
    outcomparepath = outdir + pdirname + 'PredictVsExperimentPTSB' + filenm + '.pdf'
    PTSBprcc = compareKaScape(PTSB, SBInfo, tilename, outcomparepath)
    outcomparepath = outdir + pdirname + 'PredictVsExperimentKaRI' + filenm + '.pdf'
    Kaprcc = compareKaConstantScape(RI, SInfo, SBInfo, tilename, outcomparepath)
    outdata=[tilename, PTSBprcc, Kaprcc]
    return outdata


def getKaScape(SInfo,SBInfo,cmarray,w,motiflen,upperconstraint,lowerconstraint,outdir,maxIter,loopnum,experimentIndex,LseqLen,lastlossnum,lossconvergeDis,date,randomN):
    n=4**motiflen*LseqLen
    model=Layer(n)
    currentrff=0
    converge=False
    outdata=[]
    losses = []
    for i in range(loopnum):
        #根据参数值，数据值即字符串中子字符串的位置获取目标值
        target=model.RjTarget(SBInfo,SInfo,w)

        #根据目标值，使用最小二乘法更新参数
        TC = TrainingClosure( model, SBInfo,SInfo,cmarray,target)
        rff, w=Train(TC,w,upperconstraint,lowerconstraint,maxIter)

        #若收敛，则退出

        if len(losses) > lastlossnum:
            lastlosses = losses[-lastlossnum:]
            print('loop lastlosses')
            print(lastlosses)
            print(np.std(lastlosses))
            if np.std(lastlosses) < lossconvergeDis:
                print('train final convergent')

                converge = True
                outdata = saveResult(w, experimentIndex, motiflen, i, maxIter, rff, outdir, TC, SInfo, SBInfo,date,randomN)
                return outdata

        else:
            currentrff = rff

    if not converge:
        outdata = saveResult(w, experimentIndex, motiflen, loopnum, maxIter, currentrff, outdir, TC, SInfo, SBInfo,date,randomN)
    return outdata





def writeOutDivInKmer(subkmer,outfilepath):
    fw = open(outfilepath, 'w')
    for i in range(len(subkmer)):
        for j in range(len(subkmer)):
            fw.write(str(subkmer[i][j]) + "\t")
        fw.write('\n')

def getdivExceptzero(inPathText,outPathText,outfile):
    inMat = readData(inPathText) #in
    outMat = readData(outPathText) #out
    inex = np.array(inMat) #in
    outcon = np.array(outMat) #out
    sm = np.divide(outcon, inex, out=np.zeros_like(outcon), where=inex != 0) #
    writeOutDivInKmer(sm,outfile)

#def KaScapeMain(date, input_path, output_path, outdirRoot,flank,randomN):
def KaScapeMain(paras):
    date, input_path, output_path, outdirRoot, flank, randomN,loopnum=paras
    outdir = outdirRoot + date + '_'+str(randomN) + '/'
    outdirRawdata=outdirRoot + date + '_'+str(randomN) + '/data/'
    try:
        cmd = 'mkdir ' + outdir
        os.system(cmd)
    except:
        pass

    try:
        cmd = 'mkdir ' + outdirRawdata
        os.system(cmd)
    except:
        pass
    experimentIndex = output_path.split('/')[-1][:-4]
    backgroundIndex = input_path.split('/')[-1][:-4]
    writefilepath = outdir + backgroundIndex + '_' + experimentIndex + 'div.txt'
    outfigurepath = outdir + backgroundIndex + '_' + experimentIndex + 'div.pdf'
    getdivExceptzero(input_path, output_path, writefilepath)
    printPic(writefilepath, outfigurepath,title=date + '_' + str(randomN) + backgroundIndex + '_' + experimentIndex)

    
    outfigurepath = outdir + backgroundIndex  + '.pdf'
    printPic(input_path, outfigurepath, title=date + '_' + str(randomN) + backgroundIndex )

    outfigurepath = outdir + experimentIndex + '.pdf'
    printPic(output_path, outfigurepath, title=date + '_' + str(randomN) +  experimentIndex)
    

    outdata=[]

    for length in range(1, randomN):
        cmpL=cmdir+str(length)+'.txt' # k, 短序列长度
        cmpN = cmdir + str(randomN) + '.txt' #L, 长序列长度，随机碱基数，可以变换的数据
        SDataOutpath = outdirRawdata+backgroundIndex+experimentIndex+'Sdata'+str(randomN)+'_'+str(length)+'.pkl'
        SBDataOutpath = outdirRawdata+backgroundIndex+experimentIndex+'SBdata' + str(randomN) + '_' + str(length) + '.pkl'
        LseqLen = randomN-length+1
    #读取有flank序列的数量以及每条序列对应的motif列表,包括了正反两链
        SInfoReader=DataReader(input_path,flank,length,cmpL,cmpN,SDataOutpath)
        SInfo=SInfoReader.read_out()
        SBInfoReader = DataReader(output_path, flank, length, cmpL, cmpN,SBDataOutpath)
        SBInfo = SBInfoReader.read_out()

        SInfo = pickle.load(open(SDataOutpath, 'rb'))
        SBInfo = pickle.load(open(SBDataOutpath, 'rb'))


        #初始化KaScape
        TFM=10**-7
        DNAM=5*10**-7
        PBScaleinit=0.1
        PBScaleMin=0
        PBScaleMax = 0.2
        Upperboud=float(inf)
        lowerbound=-28 #假设结合常数为pmol级别 -log(10**12)=-27.6

        maxIter = 150

        lastlossnum = 10
        lossconvergeDis = 1e-10


        upperconstraint=list(ones(4**length*LseqLen)*Upperboud)+[-13,PBScaleMax] # 化学势，假设umol级别的蛋白浓度都没有结合，log(10**(-6))=-13.8
        lowerconstraint = list(ones(4 ** length*LseqLen ) * lowerbound) + [float(-inf),PBScaleMin]

        cmarray,winit=initKaScape(cmpL,length,TFM,DNAM,PBScaleinit,LseqLen)


        res=getKaScape(SInfo,SBInfo,cmarray,winit,length,upperconstraint,lowerconstraint,outdir,maxIter,loopnum,experimentIndex,LseqLen,lastlossnum,lossconvergeDis,date,randomN)
        outdata.append(res)
        print(randomN)
        print(length)


        #break
    outccpath = outdir + date + '_' + str(randomN) + experimentIndex + '_pearsonCorrelationCoefficient.xlsx'
    pddata = pd.DataFrame(outdata, columns=['filename', 'PTSBpearsonCorrelationCoefficient','RIpearsonCorrelationCoefficient'])
    pddata.to_excel(outccpath)







def mkdir(filepath):
    try:
        os.mkdir(filepath)
    except:
        pass










if __name__ == '__main__':
    rootdir='/PROJ4/chenhong/kscape/data/PortionKmer/kmer/'
    flankdict={'1':['GCGCT', 'AGGAGTGGGATCCGGGGGGGG'],'2':['ACTCAGTG','CTAGTACGAGGAGATCTGCATCTC'],'3':['CCTGCTTTCTCGTA','AGCTCGATCTCGCACTCAGTAC']}


    cmdrootdir = '/PROJ4/chenhong/kscape/KaScapeFit/'
    cmdir = cmdrootdir + 'data/cMatrix/'

    outdirbase = cmdrootdir+'data/LCEScapeFitMainODI/'
    try:
        os.mkdir(outdirbase)
    except:
        pass

    datarootdirpath = '/PROJ4/chenhong/kscape/kascapeProcessData/all/20220715/NPortionKmer/'

    date = '20220715'
    paras = []
    loopnum=10
    fixtype = str(1)


    for i in range(4):
        randomN=i+4
        name='N'+str(randomN)
        input_path = datarootdirpath + 'I'+name+ '.txt'
        output_path = datarootdirpath + 'O'+name + '.txt'
        flank = flankdict[fixtype]
        outdir = outdirbase + '_'.join([date, name,str(loopnum)]) + '/'
        mkdir(outdir)
        paras.append([date, input_path, output_path, outdir, flank, randomN,loopnum])

    print(paras)
    with Pool(processes=20) as pool:
        pool.map(KaScapeMain, paras)

    print('ok')







