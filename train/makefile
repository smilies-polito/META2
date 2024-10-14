
dataset/%/:
	python train.py makeversion $*

dataset/%/train_functions.pickle: dataset/%/
	python train.py maketrainfunctions $*

dataset/%/test_functions.pickle: dataset/%/
	python train.py maketestfunctions $*

dataset/%/train_scores.pickle: dataset/%/train_functions.pickle
	python train.py maketrainscores $*

dataset/%/test_scores.pickle: dataset/%/test_functions.pickle
	python train.py maketestscores $*

