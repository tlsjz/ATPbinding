import pickle
import linecache
import math
import os

class PSSMfeature():
    def runblast(self,fastapath,outpath,pssmpath):
        names=[name for name in os.listdir(fastapath) if os.path.isfile(os.path.join(fastapath+'\\', name))]
        for each_item in names:
            uniprotid=each_item.split('.')[0]
            cmd='C:\\blast\\bin\\psiblast -evalue 10 -num_iterations 3 -db C:\\blast\\db\\uniprot -query '+fastapath+'\\'+each_item+' -outfmt 0 -out '+outpath+'\\'+uniprotid+'.fm0 -out_ascii_pssm '+pssmpath+'\\'+uniprotid+'.pssm -num_alignments 1500 -num_threads 8'
            #print(cmd)
            os.system(cmd)        
    def PSSMparser(self,pssmpath,pdbid):
        filelines = linecache.getlines(pssmpath+'\\'+pdbid+'.pssm')
        pssmDic = {}
        for line in filelines:
            content = line.split()
            if len(content) == 44:
                residuePosition = int(content[0])-1
                pssmDic[residuePosition] = []
                for i in range(2,22):
                    #pssmDic[str(residuePosition)].append(int(content[i]))
                    pssmDic[residuePosition].append(self.normalize(int(content[i])))
        return pssmDic
    
    def normalize(self,value):
        a = 1+math.exp(value)
        b = 1/a
        return b    
    
    def run(self,fastapath,outpath,pssmpath,pssmfeaturepath):
        self.runblast(fastapath, outpath, pssmpath)
        fastalist = os.listdir(fastapath)
        for eachfasta in fastalist:
            pdbid = eachfasta.split('.')[0]
            pssmdic = self.PSSMparser(pssmpath, pdbid)
            pssmpickle = open(pssmfeaturepath+'\\'+pdbid+'.pickle','wb')
            pickle.dump(pssmdic,pssmpickle)

if __name__=="__main__":
    pssm = PSSMfeature()
    pssm.run('D:\\atpbinding\\atp227\\fasta','D:\\atpbinding\\atp227\\blastout\\out','D:\\atpbinding\\atp227\\blastout\\pssm','D:\\atpbinding\\atp227\\pssmfeature')
        