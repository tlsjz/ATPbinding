A Novel Prediction Method for ATP-binding Sites from Protein Primary Sequences Based on Fusion of Deep Convolutional Neural Network and Ensemble Learning
Author: Jiazhi Song
Date: December 16, 2019

Hello Friends,
Please follow the following steps to execute the tool:
Description:
This program is using deep convolutional neural network and ensemble learning to identify protein-ATP binding sites based sequence information

How to use:
1.Users need to install the following tools:
  (1)Blast+ tool(for executing psiblast: sudo apt-get install ncbi-blast+ )
  (2)Psipred tool for secondary structure prediction which is included in $HOME/psipred.4.02
  (3)ASAquick tool for solvent accessibility prediction which is included in $HOME/GENN+ASAquick
  (4)Tensorflow/Keras for loading the deep network pretrained model purpose.
  
2.To run the script, some parameters needs to be modified to the specific path on your system. Use vim or any other text editor to open executable.py script file and change following paths to suit your system:
  (1)fastapath: your input fasta file path(recommended as $HOME/tool/fasta).
  (2)pdbid: protein id for fasta file.
  (3)psiblastoutpath: path for the output file for PSI-Blast(recommeded as $HOME/tool/blastout/out).
  (4)PSSMpath: path for PSSM profile(recommended as $HOME/tool/blastout/pssm).
  (5)psipredoutpath: path for output file of psipred(recommeded as $HOME/tool).
  (6)ASAquickoutpath: path for output file of ASAquick(recommended as $HOME/tool).
  (7)referenced database: $HOME/tool/blast/db/uniprot for ensemble predictor.
                          $HOME/tool/blast/db/template for sequence template-based predictor.
  (8)binding sites file of training sites as referenced sites in sequence template-based predictor: $HOME/tool/sitedic.pickle
  (9)output file path: path for the output file of the program.
  
3.run the script using following command:
  >cd $HOME/tool
  >python3 executable.py -f $HOME/tool/fasta -i 3BU_D(example)
  
4.about the output file:
  there three colums in the output file including the residue id, prediction probability from ensemble predictor and prediction result from sequence template-based predictor. The residue id refers to
the location of query residue in the input sequence, prediction probability refers to the probability of query residue classified as ATP-binding by the ensemble predictor and template-based prediction 
result is a binary prediction 
  
Thank you and wish our work can be helpful for you!
if you have any questions or suggestions, please contact:
songjz671@nenu.edu.cn