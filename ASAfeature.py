import os
import pickle
import linecache

class ASAquick():
    def runASAquick(self,fastapath):
        fastalist = os.listdir(fastapath)
        for eachfasta in fastalist:
            #pdbid = eachfasta.split('.')[0]
            cmd = '/home/songjiazhi/GENN+ASAquick/bin/ASAquick '+fastapath+'/'+eachfasta
            os.system(cmd)
            
    def ASAquickParser(self,fastapath,ASAquickPath):
        fastalist = os.listdir(fastapath)
        for eachfasta in fastalist:
            pdbid = eachfasta.split('.')[0]
            ASAquickDic = {}
            filelines = linecache.getlines(ASAquickPath+'/asaq.'+eachfasta+'/asaq.pred')
            for line in filelines:
                resid = int(line.split()[0])-1
                resname = line.split()[1]
                resvalue = float(line.split()[2])
                if resname != '-':
                    ASAquickDic[resid] = resvalue
            picklefile = open('/home/songjiazhi/atpbinding/atp227/ASAquickfeature/'+pdbid,'wb')
            pickle.dump(ASAquickDic,picklefile)
            
    def run(self,fastapath,ASAquickPath):
        self.runASAquick(fastapath)
        self.ASAquickParser(fastapath, ASAquickPath)
        
if __name__=="__main__":
    test = ASAquick()
    test.run('/home/songjiazhi/atpbinding/atp227/fasta','/home/songjiazhi/atpbinding/atp227')
            