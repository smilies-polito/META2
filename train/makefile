#Create new version
dataset/%/:
	python3 train.py makeversion $*

#Define training set functions
dataset/%/train_functions.pickle: | dataset/%/
	python3 train.py maketrainfunctions $*

dataset/%/test_functions.pickle: | dataset/%/
	python3 train.py maketestfunctions $*

#Produce raw fitness scores
dataset/%/train_scores_raw.pickle: | dataset/%/train_functions.pickle
	python3 train.py maketrainrawscores $*

dataset/%/test_scores_raw.pickle: | dataset/%/test_functions.pickle
	python3 train.py maketestrawscores $*

#Produce normalized fitness scores
dataset/%/train_scores.pickle: | dataset/%/train_scores_raw.pickle
	python3 train.py maketrainscores $*

dataset/%/test_scores.pickle: | dataset/%/test_scores_raw.pickle
	python3 train.py maketestscores $*

#Produce FLA measures
dataset/%/fla.pickle: | dataset/%/train_functions.pickle
	python3 train.py makefla $*

dataset/%/fla_test.pickle: | dataset/%/test_functions.pickle
	python3 train.py makeflatest $*

#Train and test

dataset/%/test_results/: dataset/%/fla_test.pickle dataset/%/test_scores.pickle
	mkdir dataset/$*/test_results/
	python3 train.py trainandtest $* > dataset/$*/test_results/logs.txt

dataset/%/test_results/plots: dataset/%/test_results/predictions.pickle
	python3 train.py ploterror $* 

%/clean_results: dataset/%/test_results/
	rm -rf dataset/$*/test_results/



#> dataset/$*/test_logs.txt

