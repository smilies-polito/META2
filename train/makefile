
dataset/%/:
	python3 train.py makeversion $*

dataset/%/train_functions.pickle: | dataset/%/
	python3 train.py maketrainfunctions $*

dataset/%/test_functions.pickle: | dataset/%/
	python3 train.py maketestfunctions $*

dataset/%/train_scores_raw.pickle: | dataset/%/train_functions.pickle
	python3 train.py maketrainrawscores $*

dataset/%/test_scores_raw.pickle: | dataset/%/test_functions.pickle
	python3 train.py maketestrawscores $*

