# ATPbindingprediction
ATPbinding is a sequence-based prediction method for protein-ATP binding sites prediction with deep convolutional neural network and ensemble learning. Two predictors are developed including a Residual-inception-based predictor and a Multi-inception-based predictor. The predicton result is the combination of two predictors along with a sequence-template predictor as a complementary method.

The including files are listed as follows:

Folder:
(1)dataset: The datasets applied in the study
(2)example: The running example for the executable file(PDB ID: 1EE1_A)
(3)GENN+ASAquick: Software that generates the solvent accessibility feature
(4)psipred.4.02: Software that generates the predicted secondary structure feature
(5)model: Trained classification model for Residual-inception-based predictor and Multi-inception-based predictor
(6)templateblastdb: Referenced database for sequence template-based predictor

File:
(1)residual-inception-training.py: Training process for Residual-inception-based predictor
(2)multi-inception-training.py: Training process for Multi-inception-based predictor
(3)executable.py: executable file to use our method for protein-ATP binding sites prediction
