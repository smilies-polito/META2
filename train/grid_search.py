import sys 
sys.path.append("../")
from regression_models.Random_forest import *
from regression_models.MLP_model import *
import numpy as np
import pickle
import copy
import matplotlib.pyplot as plt
import numpy as np

def build_dataset(scores, fla_measures):
    #fla: [fla_measures for f in functions]
    #scores: [[score for f in functions] for a in algorithm]
    #Training dataset: [(FLA, scores for each algorithm) for each function]
    x = np.array(fla_measures)
    y = np.array([[scores[a][i] for a in range(len(scores))] for i in range(len(scores[0]))])
    return x, y

def run_with_kfold(model, x, y, K=5):
    fold_size = x.shape[0]//K
    columns = np.arange(x.shape[0])//fold_size
    error = 0
    for i in range(K):
        x_train, y_train = copy.deepcopy(x[columns!=i,:]), copy.deepcopy(y[columns!=i,:])
        x_test, y_test = copy.deepcopy(x[columns==i,:]), copy.deepcopy(y[columns==i,:])
        m = model()
        _ = m.train(x_train, y_train)
        _, e = m.test(x_test, y_test)
        error += np.mean(e)
    return error/K

def test_random_forest(x, y, K=5):
    results = []
    i=0
    for n_estimators in [80, 100, 110, 120]:
        for min_samples_split in [2, 3, 4, 5]:
            for max_features in [40, 70,90,110, 200, "sqrt", 1]:
                print(i, 4*4*7)
                m  = lambda: RandomForest_Model(n_estimators=n_estimators, min_samples_split=min_samples_split,max_features=max_features,bootstrap=True,convert_dtype=True)
                error = run_with_kfold(m, x, y, K)
                results.append(
                    {"n_estimators":n_estimators, "min_samples_split":min_samples_split,"max_features":max_features, "error":error}
                )
                i+=1
    return results

def test_fcNN(x, y, K=5):
    results = []
    i=0
    for MAX_ITER in [3000, 3500]:
        for alpha in [0.5, 0.1, 0.01, 0.001]:
            for hidden_layer_sizes in [(20,10,10,10),(30, 30, 10, 10),(50, 50, 30, 10),(80,50,30,10), (100, 80, 30, 10),(200,100,40,10)]:
                print(i, 2*4*8)
                m = lambda: MLP_Model(alpha=alpha, hidden_layer_sizes=hidden_layer_sizes,max_iter=MAX_ITER,convert_dtype=True)
                error = run_with_kfold(m, x, y, K)
                results.append(
                    {"alpha":alpha, "hidden_layer_sizes":str(hidden_layer_sizes),"max_iter":MAX_ITER, "error":error}
                )
                i += 1
    return results



def plot_heatmap(X,Y,grid,title,save_path, xLabel, yLabel,aspect="auto"):
    # Example data (replace with your actual data)
    grid = np.array(grid)
    plt.figure(figsize=(12, 6))
    heatmap = plt.imshow(grid, cmap="viridis", aspect=aspect)
    # Set axis labels
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.xticks(ticks=np.arange(len(X)), labels=X)
    plt.yticks(ticks=np.arange(len(Y)), labels=Y)
    # Add color bar for error values
    cbar = plt.colorbar(heatmap)
    cbar.set_label("Error")
    for i in range(len(Y)):
        for j in range(len(X)):
            plt.text(j, i, f"{grid[i, j]:.4f}", ha="center", va="center", color="white")
    plt.title(title)
    plt.savefig(save_path)

def filter_results(results, filters):
    return min(map(
        lambda f: f["error"],# [e["error"] for e in f],
        filter(lambda v: all([v[k]==filters[k] for k in filters.keys()]), results)
    ))

def plot_results_fcNN(path):
    with open(f"{path}/fcNN_grid_search.pickle", "rb") as f:
        r = pickle.load(f)
    alpha_values = [0.5, 0.1, 0.01, 0.001]
    hidden_layer_size_values = [(20,10,10,10),(30, 30, 10, 10),(50, 50, 30, 10),(80,50,30,10), (100, 80, 30, 10),(200,100,40,10)]
    grid = [
        [filter_results(r, {"alpha": alpha, "hidden_layer_sizes": str(l)}) for alpha in alpha_values]
        for l in hidden_layer_size_values
    ]
    plot_heatmap(alpha_values, hidden_layer_size_values,grid,"Validation error for fcNN", f"{path}/fcNN_heatmap.png", "Alpha","Layers size",aspect=0.5)

def plot_results_RF(path):
    with open(f"{path}/RF_grid_search.pickle", "rb") as f:
        r = pickle.load(f)
    n_estimators = [80, 100, 110, 120]
    min_samples_split = [2, 3, 4, 5]
    max_features = [40, 70,90,110, 200, "sqrt", 1]
    
    grid = [
        [filter_results(r, {"n_estimators": n_e, "min_samples_split": m_s_s}) for n_e in n_estimators]
        for m_s_s in min_samples_split
    ]
    plot_heatmap(n_estimators, min_samples_split,grid,"Validation error for RF", f"{path}/RF_heatmap_1.png", "N. Estimators","Min S. Split")
    
    grid = [
        [filter_results(r, {"n_estimators": n_e, "max_features": mf}) for n_e in n_estimators]
        for mf in max_features
    ]
    plot_heatmap(n_estimators, max_features,grid,"Validation error for RF", f"{path}/RF_heatmap_2.png", "N. Estimators","Max Features")
    
    grid = [
        [filter_results(r, {"min_samples_split": m_s_s, "max_features": mf}) for m_s_s in min_samples_split]
        for mf in max_features
    ]
    plot_heatmap(min_samples_split, max_features,grid,"Validation error for RF", f"{path}/RF_heatmap_3.png", "Min S. Split","Max Features")
    

def load_data(path):
    #Load FLA measures
    with open(f"{path}/fla_test.pickle", "rb") as f:
        fla_train, fla_test = pickle.load(f)
    #Load scores
    with open(f"{path}/test_scores.pickle", "rb") as f:
        scores_train, scores_test = pickle.load(f)
    #Build dataset
    x_train, y_train = build_dataset(scores_train, fla_train)
    x_test, y_test = build_dataset(scores_test, fla_test)
    return x_train, y_train, x_test, y_test

def run_grid_search(output_dir):
    x_train, y_train, _, _ = load_data(output_dir)
    rf_result = test_random_forest(x_train, y_train, 5)
    with open(f"{output_dir}/RF_grid_search.pickle", "wb") as f:
        pickle.dump(rf_result,f)
    fcNN_result = test_fcNN(x_train, y_train, 5)
    with open(f"{output_dir}/fcNN_grid_search.pickle", "wb") as f:
        pickle.dump(fcNN_result,f)

run_grid_search("dataset/v6")
plot_results_RF("dataset/v6")
plot_results_fcNN("dataset/v6")