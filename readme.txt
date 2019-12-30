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
  (2)The SwissProt Database that is used for running PSI-BLAST can be available at ftp://ftp.uniprot.org/pub/databases/uniprot/knowledgebase
  (3)Psipred tool for secondary structure prediction which is included in $HOME/ATPbinding/psipred.4.02
  (4)ASAquick tool for solvent accessibility prediction which is included in $HOME/ATPbinding/GENN+ASAquick
  (5)Tensorflow/Keras for loading the deep network pretrained model purpose.
  
2.To run the script, some parameters needs to be modified to the specific path on your system. Use vim or any other text editor to open executable.py script file and change following paths to suit your system:
  (1)fastapath: your input fasta file path(recommended as $HOME/ATPbinding/example/fasta).
  (2)pdbid: protein id for fasta file.
  (3)psiblastoutpath: path for the output file for PSI-Blast(recommeded as $HOME/ATPbinding/example/blastout/out).
  (4)PSSMpath: path for PSSM profile(recommended as $HOME/ATPbinding/example/blastout/pssm).
  (5)psipredoutpath: path for output file of psipred(recommeded as $HOME/ATPbinding/example).
  (6)ASAquickoutpath: path for output file of ASAquick(recommended as $HOME/ATPbinding/example).
  (7)referenced database: $HOME/ATPbinding/blast/db/uniprot for ensemble predictor.
                          $HOME/ATPbinding/templateblastdb/template for sequence template-based predictor.
  (8)When you use the trained model, you may need to decompress the model file via $rar e $Home/ATPbinding/model/ResidualInception.part1.rar and $rar e $Home/ATPbinding/model/MultiInception.part1.rar
  (9)binding sites file of training data as referenced sites in sequence template-based predictor: $HOME/ATPbinding/sitedic.pickle
  (10)output file path: path for the output file of the program.
  
3.run the script using following command:
  >cd $HOME/ATPbinding
  >python3 executable.py -f $HOME/ATPbinding/example/fasta -i 1EE1_A(example)
  
4.about the output file:
  there are three colums in the output file including the residue id, prediction probability from ensemble predictor and prediction result from sequence template-based predictor. The residue id refers to
the location of query residue in the input sequence, prediction probability refers to the probability of query residue classified as ATP-binding by the ensemble predictor and template-based prediction 
result is a binary prediction 
  
The code is written and tested on Ubuntu 16.04.5
Thank you and wish our work can be helpful for you!
if you have any questions or suggestions, please contact:
songjz671@nenu.edu.cn