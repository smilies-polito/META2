import sys
import math
import numpy as np
import os
import pickle
from problems.BaseProblem import BaseProblem
from solvers.ga import GA
from solvers.de import DE
from solvers.pso import PSO
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from rpy2.robjects import r, numpy2ri

class FEM:
    #Can be interesting to compute FEM for both binary and continuous strategies for the random walk
    def bitwise_random_walk(N: int, d: int) -> np.array:
        pass
    def continuous_random_increasing_walk(problem: BaseProblem, N: int, step_size: float = 1/100) -> np.array:
        step_size = (problem.get_ranges()[0][1] - problem.get_ranges()[0][0]) * step_size
        random_walk_states = np.zeros((N, len(problem.get_ranges())))
        random_walk_states[0] = problem.sample_uniform(1)[0]
        for i in range(N):
            if (i == N-1): break
            random_walk_states[i+1] = random_walk_states[i] + np.random.uniform(-step_size, step_size, problem.get_dimensions())

        random_walk = np.array([
                problem.get_value(random_walk_states[i]) for i in range(N)
            ])
        return np.array(random_walk), random_walk_states
            

    def psi_function(random_walk: np.array, epsilon: float) -> np.array:
        # -1: decrease, 0: stay, 1: increase - epsilon is the sensitivity
        return np.array([
            -1 if (x-random_walk[i-1]) < -epsilon else 0 if (x-random_walk[i-1]) <= epsilon else 1
            for i,x in enumerate(random_walk[1:])
        ])
    
    def entropic_measure(psi: np.array) -> float:
        subblocks = {"01": 0, "0-1": 0, "10": 0, "1-1": 0, "-10": 0, "-11":0}
        for i, x in enumerate(psi[1:]):
            if (x == psi[i-1]): continue 
            key = f"{psi[i-1]}{x}"
            subblocks[key] += 1
        for key in subblocks:
            subblocks[key] /= len(psi)
        entropy = 0
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if (i==j): continue 
                pi = subblocks[f"{i}{j}"]
                if (pi>0):
                    entropy += pi * math.log(pi,6)
        return -entropy

class FDC:
    def get_FDC(problem: BaseProblem, sampled_points: np.array, fitness_values: np.array):
        def compute_FDC(sampled_points, fitness_values):
            best_point = sampled_points[np.argmin(fitness_values)]
            distance_vector = np.array([np.linalg.norm(best_point - point) for point in sampled_points])
            #Compute covariance between fitness_values and distance_vector
            covariance = np.cov(fitness_values, distance_vector)[0,1]
            return covariance / (np.std(fitness_values) * np.std(distance_vector))

        #Compute global metric with all sampled points
        global_metric = compute_FDC(sampled_points, fitness_values)
        #Local metrics: split sampled points into 4 groups 
        groups = []
        for i, sampled_point in enumerate(sampled_points):
            local_group = groups
            for dimension in range(problem.get_dimensions()):
                if len(local_group) == 0:
                    local_group += [[],[]]
                if (sampled_point[dimension] < problem.get_ranges()[dimension][0] + (problem.get_ranges()[dimension][1] - problem.get_ranges()[dimension][0])/2):
                    local_group = local_group[0] 
                else:
                    local_group = local_group[1]
            local_group.append(i)
        #Compute metrics separately for each group
        local_FDC_metrics = []
        def local_metrics_recursive(groups, dimension_index):
            if (dimension_index == 0):
                if (len(groups) == 0): return
                local_sampled_points = sampled_points[groups]
                local_fitness_values = fitness_values[groups]
                if (len(local_sampled_points) > 2):
                    v = compute_FDC(local_sampled_points, local_fitness_values)
                    if (not np.isnan(v)):
                        local_FDC_metrics.append(v)
            elif (len(groups) > 1):
                local_metrics_recursive(groups[0], dimension_index-1)
                local_metrics_recursive(groups[1], dimension_index-1)
        local_metrics_recursive(groups, problem.get_dimensions())
        fdc_avg, fdc_stdev, fdc_min, fdc_max = np.mean(local_FDC_metrics), np.std(local_FDC_metrics), np.min(local_FDC_metrics), np.max(local_FDC_metrics)
        return [global_metric, fdc_avg, fdc_stdev, fdc_min, fdc_max]
        
class DispersionMetric:
    def get_dispersion_metrics(sampled_points: np.array, fitness_values: np.array, percentage = [0.005, 0.1, 0.2, 0.4]):
        #Normalize sampled_points
        sampled_points = (sampled_points - np.mean(sampled_points, axis=0)) / np.std(sampled_points, axis=0)
        #Compute the dispersion of sampled_points, as average of pair-wise distance
        dispersion = 0
        N = len(sampled_points)
        for i in range(N):
            for j in range(i+1, N):
                dispersion += np.linalg.norm(sampled_points[i] - sampled_points[j])
        dispersion /= N*(N-1)/2
        results = []
        for p in percentage:
            #Get indexes of the best "percentage" points of fitness_values
            best_points = np.argsort(fitness_values)[:int(N*p)]
            #Compute the dispersion of the best points
            best_points_dispersion = 0
            for i in range(len(best_points)):
                for j in range(i+1, len(best_points)):
                    best_points_dispersion += np.linalg.norm(sampled_points[best_points[i]] - sampled_points[best_points[j]])
            best_points_dispersion /= len(best_points)*(len(best_points)-1)/2
            results.append(best_points_dispersion - dispersion)
        return results

class JensensInequalityRatio:
    def get_jensen_inequality_ratio(problem: BaseProblem, sampled_points: np.array, fitness_values: np.array, N: int):
        count = 0
        for _ in range(N):
            i, j = np.random.choice(len(sampled_points), 2, replace=False)
            middle_point = problem.get_value((sampled_points[i] + (sampled_points[j]))/2)
            if (middle_point < (fitness_values[i] + fitness_values[j])/2):
                count += 1
        return count/N

class FitnessEstimator:
    def get_fitness_est_measure(sampled_points: np.array, fitness_values: np.array):
        fitness_var = np.std(fitness_values)**2
        #Fit a linear model to estimate the fitness values
        linear_err = mean_squared_error(fitness_values, LinearRegression().fit(sampled_points, fitness_values).predict(sampled_points)) / fitness_var
        #Now try to fit linear model to half of the dataset and test on the rest
        half = int(len(fitness_values)/2)
        linear_err_val = mean_squared_error(fitness_values[:half], LinearRegression().fit(sampled_points[half:,:], fitness_values[half:]).predict(sampled_points[half:,:])) / fitness_var
        linear_err_val += mean_squared_error(fitness_values[half:], LinearRegression().fit(sampled_points[:half,:], fitness_values[:half]).predict(sampled_points[:half,:])) / fitness_var
        linear_err_val /= 2
        #Now fit N linear models for the N variables
        single_v_models_lin = [
            LinearRegression().fit(sampled_points[:,i].reshape(-1, 1), fitness_values)
            for i in range(sampled_points.shape[1])]
        single_v_predictions = np.sum([
            m.predict(sampled_points[:,i].reshape(-1, 1)) for i, m in enumerate(single_v_models_lin)
            ], axis=0)
        single_v_predictions += np.mean(fitness_values) - np.mean(single_v_predictions)
        single_v_prediction_errors_lin = mean_squared_error(fitness_values,single_v_predictions) / fitness_var

        #Fit a random forest to estimate the fitness values
        model = RandomForestRegressor()
        model.fit(sampled_points, fitness_values)
        prediction_error = mean_squared_error(fitness_values, model.predict(sampled_points)) / fitness_var

        #Now try to fit model to half of the dataset and test on the rest
        half = int(len(fitness_values)/2)
        err_val = mean_squared_error(fitness_values[:half], RandomForestRegressor().fit(sampled_points[half:,:], fitness_values[half:]).predict(sampled_points[half:,:])) / fitness_var
        err_val += mean_squared_error(fitness_values[half:], RandomForestRegressor().fit(sampled_points[:half,:], fitness_values[:half]).predict(sampled_points[:half,:])) / fitness_var
        err_val /= 2

        #Now fit N models for the N variables
        single_v_models = [
            RandomForestRegressor().fit(sampled_points[:,i].reshape(-1, 1), fitness_values)
            for i in range(sampled_points.shape[1])]
        single_v_prediction_errors = mean_squared_error(fitness_values, np.average([
            m.predict(sampled_points[:,i].reshape(-1, 1)) for i, m in enumerate(single_v_models)
            ], axis=0)) / fitness_var
        #Return two measures: accuracy of the model and the ratio of the single variable models
        return [linear_err,linear_err_val, single_v_prediction_errors_lin/linear_err , prediction_error, err_val, single_v_prediction_errors / prediction_error]

class FlaccoFLA:
    def get_features(sampled_points: np.array, fitness_values: np.array):
        result = r.flacco_FLA(sampled_points, fitness_values)
        result = np.array([r[0] for r in result])
        #set all NaNs to 0
        result[np.isnan(result)] = 0
        result[np.isinf(result)] = sys.float_info.max
        return result

class NonDeterminismFLA:
    def get_nd_measures(problem: BaseProblem, sampled_points: np.array, fitness_values: np.array):
        #Sample 20 points
        indexes = np.random.choice(len(sampled_points), 20)
        sampled_points = sampled_points[indexes]
        fitness_values = fitness_values[indexes]
        stds = []
        stds_norm = []
        #For each point:
        for i, point in enumerate(sampled_points):
            #1-Measure of std with 5 samples
            f = [problem.get_value(point) for _ in range(4)] + [fitness_values[i]]
            std = np.std(f)
            #2-Measure stdev / (mean)
            if (np.mean(f) == 0):
                std_mean = std
            else:
                std_mean = std / np.mean(f)
            stds.append(std)
            stds_norm.append(std_mean)
        #3-Return mean of std, std of std, mean of normalized std, std of normalized std
        return [np.mean(stds), np.std(stds), np.mean(stds_norm), np.std(stds_norm)]

class SimpleEvolvability:
    def get_evolvability(sampled_points, fitness_values, Ms=[2,4,6,8]):
        #Compute distance matrix
        distance_matrix = np.zeros((sampled_points.shape[0], sampled_points.shape[0]))
        for i in range(sampled_points.shape[0]-1):
            for j in range(i+1, sampled_points.shape[0]):
                d = np.linalg.norm(sampled_points[i] - sampled_points[j])
                distance_matrix[i][j] = d 
                distance_matrix[j][i] = d 

        evolvabilities = []
        LPPs = []

        for index,M in enumerate(Ms):
            evolvabilities.append([])
            LPPs.append([])
            for i in range(sampled_points.shape[0]):
                closest_points = np.argsort(distance_matrix[i])[:M]
                closest_fitness = fitness_values[closest_points]
                point_fitness = fitness_values[i]
                evolvability = np.sum(closest_fitness > point_fitness)/M
                evolvabilities[index].append(evolvability)
                LPP = ""
                for j in range(M):
                    if closest_fitness[j] > point_fitness:
                        LPP += "1"
                    else:
                        LPP += "0"
                LPP = int(LPP,2)
                LPPs[index].append(LPP)
        evolvabilities = [np.mean(evolvabilities[i]) for i in range(len(Ms))] + [np.std(evolvabilities[i]) for i in range(len(Ms))] 
        LPPs = [np.mean(LPPs[i]) for i in range(len(Ms))] + [np.std(LPPs[i]) for i in range(len(Ms))]
        return evolvabilities + LPPs


    def get_evolvability_ga(sampled_points, fitness_values, problem, repeat_n):
        EPP = 0
        EAP = 0
        PERCENT_BETTER = 0
        for _ in range(repeat_n):
            c = np.random.choice(sampled_points.shape[0], 16, replace=False)
            sampled_population, population_fitness = np.array(sampled_points[c]), np.array(fitness_values[c])
            best_fitness = np.max(-population_fitness)
            next_gen = GA.solve_with_starting_population(sampled_population, problem)
            if (np.max(next_gen) > best_fitness):
                EPP += 1
            EAP += np.mean(next_gen) + np.mean(population_fitness)
            PERCENT_BETTER = len(next_gen[next_gen>best_fitness])/len(next_gen)
        EPP/=repeat_n
        EAP/=repeat_n
        PERCENT_BETTER/=repeat_n
        return [EPP, EAP, PERCENT_BETTER]

    def get_evolvability_de(sampled_points, fitness_values, problem, repeat_n):
        EPP = 0
        EAP = 0
        PERCENT_BETTER = 0
        for iteration in range(repeat_n):
            c = np.random.choice(sampled_points.shape[0], 16, replace=False)
            sampled_population, population_fitness = np.array(sampled_points[c]), np.array(fitness_values[c])
            best_fitness = np.min(population_fitness)
            next_gen = DE.solve_with_starting_population(sampled_population, problem)
            if (np.min(next_gen) < best_fitness):
                EPP += 1
            EAP += np.mean(population_fitness) - np.mean(next_gen)
            PERCENT_BETTER = len(next_gen[next_gen<best_fitness])/len(next_gen)
        EPP/=repeat_n
        EAP/=repeat_n
        PERCENT_BETTER/=repeat_n
        return [EPP, EAP, PERCENT_BETTER]

class FLA:
    r2py_active = False
    def get_FLA_measures(problem: BaseProblem, sample_size, FEM_params, jensens_inequality_N, NON_DETERMINISTIC=False):
        #Activate r2py
        if not FLA.r2py_active:
            numpy2ri.activate()
            r.source('fla/flacco_FLA.r')
            r2py_active = True
        #Generate dataset
        sampled_points = problem.sample_uniform(sample_size)
        fitness_values = np.array([problem.get_value(point) for point in sampled_points])
        random_walks = []
        for _ in range(FEM_params["repeat_random_walk"]):
            for r_w in FEM_params["random_walks"]:
                random_walk, random_walk_states = FEM.continuous_random_increasing_walk(problem, r_w["random_walk_len"], r_w["step_size"])
                random_walks.append(random_walk)
                sampled_points = np.vstack((random_walk_states, sampled_points))
                fitness_values = np.concatenate((random_walk, fitness_values))
        
        #Compute FLA measures
        measures = []
        if NON_DETERMINISTIC:
            measures += NonDeterminismFLA.get_nd_measures(problem, sampled_points, fitness_values)
        #Dimensionality measure
        measures.append(len(problem.get_ranges()))
        #FEM measure
        for random_walk in random_walks:
            measures += [
                FEM.entropic_measure(FEM.psi_function(random_walk, epsilon))
                for epsilon in FEM_params["epsilon"]
            ]
        #Dispersion measure
            measures += DispersionMetric.get_dispersion_metrics(sampled_points, fitness_values, [0.005, 0.1, 0.2, 0.4, 0.5, 0.7])
        #FDC measure
        measures += FDC.get_FDC(problem, sampled_points, fitness_values)
        #Jensen's inequality ratio
        measures.append(JensensInequalityRatio.get_jensen_inequality_ratio(problem, sampled_points, fitness_values, jensens_inequality_N))
        measures += FitnessEstimator.get_fitness_est_measure(sampled_points, fitness_values)
        #Evolvability
        measures += SimpleEvolvability.get_evolvability(sampled_points, fitness_values)
        measures += [np.percentile(fitness_values, 10),np.percentile(fitness_values, 90)]
        measures += SimpleEvolvability.get_evolvability_ga(sampled_points, fitness_values, problem, 10)
        measures += SimpleEvolvability.get_evolvability_de(sampled_points, fitness_values, problem, 9)
        #measures += PSO.local_vs_globalperformance(problem)
        #Flacco
        measures = np.concatenate((np.array(measures),FlaccoFLA.get_features(sampled_points, fitness_values)))
        return measures