import os
import linecache
import pickle

class psipredparser():
    def runpsipred(self,fastapath):
        cmd = '/home/songjiazhi/psipred.4.02/psipred/BLAST+/runpsipredplus '+fastapath
        os.system(cmd)        
    def singleparser(self,filepath):
        psipredDic = {}
        filelines = linecache.getlines(filepath)
        length = len(filelines)
        for i in range(2,length):
            residuenum = int(filelines[i].split()[0])-1
            psipredDic[residuenum] = []
            psipredDic[residuenum].append(float(filelines[i].split()[3]))
            psipredDic[residuenum].append(float(filelines[i].split()[4]))
            psipredDic[residuenum].append(float(filelines[i].split()[5]))
        return psipredDic
    
    def constructfeature(self,fastapath):
        fastalist = os.listdir(fastapath)
        for eachfasta in fastalist:
            self.runpsipred(fastapath+'/'+eachfasta)
            pdbid = eachfasta.split('.')[0]
            psipredDic = self.singleparser('/home/songjiazhi/atpbinding/atp227/psipredout/'+pdbid+'.ss2')
            picklefile = open('/home/songjiazhi/atpbinding/atp227/psipredfeature/'+pdbid+'.pickle','wb')
            pickle.dump(psipredDic,picklefile)
            
if __name__=="__main__":
    test = psipredparser()
    #test.singleparser('C:\\Users\\admin\\Desktop\\psipredout\\1A0I_A.ss2')
    test.constructfeature('/home/songjiazhi/atpbinding/atp227/fasta')