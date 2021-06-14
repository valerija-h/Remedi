The following things are important to note:
- The code was constructed and run on a macOS system.
- You will need to create a python environment with python version 3.6 and import the 'environment.yml' file. In our case, this was done using Anaconda. If the import does not work, try adding the libraries manually (the versions of Keras and Tensorflow are important) through pip and then adding the external libraries in 'Libraries/' through the instructions below.
- The contents of the folders in 'Dataset/' were cleared as it may contain traces of sensitive information. To retrieve the contents, get access to use the dataset from the following link: https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/. Once access is granted put the training and testing files into their respective directories. The folder structure was left the same so that the files are put in the correct places. (For eg. the relation annotations of the testing set should be in 'Dataset/test/rel').
- The contents of the folder 'Preprocessed Dataset/' were removed as it may contain traces of sensitive information. To acquire these files, run the code in 'Preprocessing Code/' starting from 'NER-preprocess.py' followed by 'RE-preprocess.py' - the order is important.
- The contents of the folder 'Libraries/public_mm/' (the MetaMap tool) were removed - please refer to the section below for more information on how to acquire it. Please note that once these contents are added and installed, the paths of the MetaMap tool in the 'Preprocessing Code/NER-preprocess.py' and 'remedi.py' may need to be changed to the exact path of where it was installed (change the path in 'mm = MetaMap.get_instance()').

How to install external libraries in the 'Libraries/' folder:
- 'geniatagger-3.0.2/' - the instructions can be found in http://www.nactem.ac.uk/GENIA/tagger/. Enter the directory through the terminal and run the command 'make'.
- 'keras-contrib/' - the instructions can be found in https://github.com/keras-team/keras-contrib. Enter the directory and run the command 'python setup.py install'.
- 'public_mm' - follow the instructions on this page; https://metamap.nlm.nih.gov/Docs/README.html. Please note that once these contents are added and installed, the paths of the MetaMap tool in the 'Preprocessing Code/NER-preprocess.py' and 'remedi.py' may need to be changed to the exact path of where it was installed (change the path in 'mm = MetaMap.get_instance()').
- 'seqeval-master' - simply run 'pip install seqeval[cpu]' or 'pip install seqeval[gpu]' depending on whether you have Tensorflow CPU or GPU. The instructions can be found here https://github.com/chakki-works/seqeval.

A rough description of code files:
- 'remedi.py' - the final Remedi system. The code extracts the sentences from 'input.txt', then loads the pre-trained final Bi-LSTM and LSTM models from 'Models/' and outputs the found entities and relations between entity pairs to the terminal.
- 'Preprocessing Code/NER-preprocess.py' and 'Preprocessing Code/RE-preprocess.py' - these files preprocess the dataset in the 'Dataset/' folder and extracts their features using the GENIA tagger and UMLS, then stores the preprocessed data in the 'Preprocessed Dataset/' folder as Pickle objects for the BM-NER and BM-RE component respectively.
- 'Training Code/training-CRF.py' - the file that was used for hyperparameter tuning of the CRF model and building the best final model.
- 'Training Code/training-Bi-LSTM.py' and 'Training Code/training-Bi-LSTM-CRF.py' - the files that were used for hyperparameter tuning of the Bi-LSTM and Bi-LSTM-CRF model.
- 'Training Code/build-final-models.py' - the file that was used to build the final models of the Bi-LSTM and Bi-LSTM-CRF models using the best-discovered hyperparameters.
