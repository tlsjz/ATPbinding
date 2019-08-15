import pickle
import os
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler

class feature():
    def ensemble(self,fastapath,pssmpath,psipredpath,ASApath,chemicalpath,featurepath):
        fastalist = os.listdir(fastapath)
        for eachfasta in fastalist:
            pdbid = eachfasta.split('.')[0]
            print(pdbid)
            pssmpickle = open(pssmpath+'\\'+pdbid,'rb')
            pssmdic = pickle.load(pssmpickle)
            length = len(pssmdic.keys())
            psipredpickle = open(psipredpath+'\\'+pdbid,'rb')
            psipreddic = pickle.load(psipredpickle)
            ASApickle = open(ASApath+'\\'+pdbid,'rb')
            ASAdic = pickle.load(ASApickle)
            chemicalpickle = open(chemicalpath+'\\'+pdbid,'rb')
            chemicaldic = pickle.load(chemicalpickle)
            featuredic = {}
            for i in range(0,length):
                featuredic[i] = []
                for each in pssmdic[i]:
                    featuredic[i].append(each)
                for each in psipreddic[i]:
                    featuredic[i].append(each)
                featuredic[i].append(ASAdic[i])
                for each in chemicaldic[i]:
                    featuredic[i].append(each)
            featurepickle = open(featurepath+'\\'+pdbid,'wb')
            pickle.dump(featuredic,featurepickle)
    
    def appendzero(self,windowsize,pssmDic):
        seqlength = len(pssmDic.keys())
        appendnum = int((windowsize+1)/2)
        for i in range(1,appendnum):
            pssmDic[0-i] = []
            pssmDic[seqlength-1+i] = []
            for a in range(31):
                pssmDic[0-i].append(0)
            for b in range(31):
                pssmDic[seqlength-1+i].append(0)
        return pssmDic
    
    def combine(self,sequencelength,pssmdic,windowsize):
        neighnum = int((windowsize-1)/2)
        combineDic = {}
        for i in range(0,sequencelength):
            combineDic[i] = []
            for a in range(i - neighnum,i + neighnum + 1):
                #combineDic[i].append(pssmdic[a])
                for each in pssmdic[a]:
                    combineDic[i].append(each)
        return combineDic
    
    def windowfeature(self,featurepath, windowsize, combinefeaturepath):
        proteinlist = os.listdir(featurepath)
        for eachprotein in proteinlist:
            pdbid = eachprotein.split('.')[0]
            featurepickle = open(featurepath+'\\'+pdbid,'rb')
            featuredic = pickle.load(featurepickle)
            sequencelength = len(featuredic.keys())
            appendedfeaturedic = self.appendzero(windowsize, featuredic)
            combinefeaturedic = self.combine(sequencelength, appendedfeaturedic, windowsize)    
            combinepickle = open(combinefeaturepath+'\\'+pdbid,'wb')
            pickle.dump(combinefeaturedic,combinepickle)
            
    def inputfeature(self,featurepath):
        labellist = []
        featurelist = []
        sitepickle = open('D:\\atpbinding\\atp227\\sitedic.pickle','rb')
        sitedic = pickle.load(sitepickle)
        proteinlist = os.listdir(featurepath)
        for eachprotein in proteinlist:
            #labellist = []
            #featurelist = []
            print(eachprotein)
            featurepickle = open(featurepath+'\\'+eachprotein,'rb')
            featuredic = pickle.load(featurepickle)
            length = len(featuredic.keys())
            for i in range(0,length):
                featurelist.append(featuredic[i])
                if i not in sitedic[eachprotein]:
                    labellist.append(0)
                else:
                    labellist.append(1)
            #data = (labellist,featurelist)
            #picklefile = open('D:\\atpbinding\\independent\\'+eachprotein,'wb')
            #pickle.dump(data,picklefile)
        data = (labellist,featurelist)
        picklefile = open('D:\\atpbinding\\atp227\\feature15.pickle','wb')
        pickle.dump(data,picklefile)
        
    def fivefold(self, traindatapath, fivefoldpath):
        traindatapickle = open(traindatapath,'rb')
        traindata = pickle.load(traindatapickle)
        feature = traindata[1]
        label = traindata[0]
        kf = StratifiedKFold(n_splits=5)
        i = 1
        for train,test in kf.split(feature,label):
            feature_train = []
            label_train = []
            feature_test = []
            label_test = []
            for each in train:
                feature_train.append(feature[each])
                label_train.append(label[each])
            for each in test:
                feature_test.append(feature[each])
                label_test.append(label[each])
            fulltraindata = (label_train, feature_train)
            fulltraindatapickle = open(fivefoldpath+'\\'+str(i)+'\\fulltrain.pickle','wb')
            pickle.dump(fulltraindata, fulltraindatapickle)
            testdata = (label_test, feature_test)
            testdatapickle = open(fivefoldpath+'\\'+str(i)+'\\test.pickle','wb')
            pickle.dump(testdata, testdatapickle)
            i = i+1
            
    
if __name__=="__main__":
    test = feature()
    test.ensemble('D:\\atpbinding\\atp227\\fasta','D:\\atpbinding\\atp227\\blastout\\pssmfeature','D:\\atpbinding\\atp227\\psipredfeature','D:\\atpbinding\\atp227\\ASAquickfeature','D:\\atpbinding\\atp227\\chemicalfeature','D:\\atpbinding\\atp227\\featureimportance\\pssmssasa\\feature')
    test.windowfeature('D:\\atpbinding\\atp227\\feature',15,'D:\\atpbinding\\atp227\\feature15')
    test.inputfeature('D:\\atpbinding\\atp227\\feature15')
    test.fivefold('D:\\atpbinding\\atp227\\feature15.pickle','D:\\atpbinding\\atp227\\fivefold')