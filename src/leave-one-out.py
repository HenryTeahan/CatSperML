import numpy as np
import argparse
import os
from joblib import delayed, Parallel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from itertools import combinations
import pandas as pd
from pdll import PairwiseDifferenceClassifier
    

def main(args):
    def predict_symmetric_pairwise(model, X_train, X_test, n_jobs=-1):
        """
        Symmetrized prediction using PDL with parallel processing.
        """
        def compute_probabilities(x_test):
            n_train = X_train.shape[0]

            z_forward = np.concatenate([
                np.tile(x_test, (n_train, 1)),
                X_train,
                np.subtract(np.tile(x_test, (n_train, 1)), X_train)
            ], axis=1)

            z_reverse = np.concatenate([
                X_train,
                np.tile(x_test, (n_train, 1)),
                np.subtract(X_train, np.tile(x_test, (n_train, 1)))
            ], axis=1)

            probs_forward = model.predict_proba(z_forward)
            probs_reverse = model.predict_proba(z_reverse)

            probs_sym = 0.5 * (probs_forward + probs_reverse)

            tree_predictions = [tree.predict(z_forward) for tree in model.estimators_]
            std = np.std(tree_predictions, axis=0)
            sig_tr = np.mean(std)
            orig_uq = np.sqrt(sig_tr**2 / (np.shape(tree_predictions)[0] * np.shape(tree_predictions)[1]))

            return probs_sym.sum(axis=0), orig_uq

        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_probabilities)(x_test) for x_test in X_test
        )

        prob_list, unc_list = zip(*results)

        prob_list = np.array(prob_list)
        prob_list = prob_list / prob_list.sum(axis=1, keepdims=True)  # Normalize
        predictions = np.argmax(prob_list, axis=1)  # axis=1 because shape is (n_test, n_classes)

        return predictions, prob_list, unc_list
    
    def predict_pairwise(model, X_train, X_test, n_jobs=-1):
        """
        Optimized version of predict_pairwise using parallel processing.
        """
        def compute_probabilities(x_test):

            z_pairs = np.concatenate([np.tile(x_test, (X_train.shape[0], 1)), 
                                       X_train, 
                                       np.subtract(x_test, X_train)], axis=1)

            probs = model.predict_proba(z_pairs)
            tree_predictions = []

            for tree in model.estimators_:
                tree_predictions.append(tree.predict(z_pairs))

            std = np.std(tree_predictions, axis=0)
            sig_tr = np.mean(std)
            orig_uq = np.sqrt(sig_tr**2/(np.shape(tree_predictions)[0]*np.shape(tree_predictions)[1]))
        
            return probs.sum(axis=0), orig_uq

        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_probabilities)(x_test) for x_test in X_test
        )
        prob_list, unc_list = zip(*results)

        prob_list = np.array(prob_list)
        prob_list = prob_list / prob_list.sum(axis=1, keepdims=True)
        predictions = np.argmax(prob_list, axis=0)

        return predictions, prob_list, unc_list

    def generate_pairwise_differences(X, y):
        """ 
        Creates pairwise differences.
        X: training data,
        y: training labels
        """
        n_samples = X.shape[0]
        idx = np.array(list(combinations(range(n_samples), 2)))  
        X_i, X_j = X[idx[:, 0]], X[idx[:, 1]]
        pairwise_X = np.hstack([X_i, X_j, X_i - X_j])
        pairwise_y = (y[idx[:, 0]] == y[idx[:, 1]]).astype(int)  

        return pairwise_X, pairwise_y
    def generate_all_differences(X, y):
        X = np.array(X)
        y = np.array(y)

        n_samples = X.shape[0]

        idx1, idx2 = np.meshgrid(np.arange(n_samples), np.arange(n_samples), indexing='ij')
        idx1 = idx1.flatten()
        idx2 = idx2.flatten()

        X_i = X[idx1]
        X_j = X[idx2]

        pairwise_X = np.hstack([X_i, X_j, X_i - X_j])

        pairwise_y = (y[idx1] == y[idx2]).astype(int)

        return pairwise_X, pairwise_y


    def train_pdl_classifier(X, y, generate_all = False):
        """
        Optimized training function for pairwise difference learning.
        """

        if generate_all:
            pairwise_X, pairwise_y = generate_all_differences(X, y)
        else:
            pairwise_X, pairwise_y = generate_pairwise_differences(X, y)

        pairwise_X = np.array(pairwise_X)
        pairwise_y = np.array(pairwise_y)
        print(np.shape(pairwise_X))

        model = RandomForestClassifier(
            n_estimators=75,
            max_depth=12,
            max_features=None,
            min_samples_leaf=2,
            min_samples_split=9,
            class_weight='balanced',
            n_jobs=4,
            random_state = 42  
        )
        model.fit(pairwise_X, pairwise_y)

        return model
    
    
    X = np.load(args.X_train)
    y = np.load(args.y)


    y = np.array(y)

    print(f"Loaded data shape: X={X.shape}, y={y.shape}")
    
    #model = train_pdl_classifier(X, y, generate_all = True)
    #predictions, probabilities, uncertainties = predict_pairwise(model, X, X, n_jobs=args.n_jobs)
    # Leave-one-out cv for pdl rf
    loo = LeaveOneOut()
    loo_probs = []
    loo_probs_sym = []
    loo_probs_n2 = []
    loo_probs_n2_sym = []
    loo_probs_full = []
    X = np.array(X); y = np.array(y)
    print(np.shape(X), np.shape(y))


    ### Beneath: NCR
    
    
    for i, (train_index, test_index) in enumerate(loo.split(X)):
        print(f"Train progress: {i / len(y)*100:.2f}")
        model = RandomForestClassifier(
            n_estimators=75,
            max_depth=12,
            max_features=None,
            min_samples_leaf=2,
            min_samples_split=9,
            class_weight='balanced',
            n_jobs=4,
        )

        model = train_pdl_classifier(X[train_index],y[train_index], generate_all = False)
        pred, probs, unc = predict_pairwise(model, X[train_index], X[test_index])
        loo_probs.append(probs)
        pred, probs, unc = predict_symmetric_pairwise(model, X[train_index], X[test_index])
        loo_probs_sym.append(probs)
        print("Finished predictions.")

    for i, (train_index, test_index) in enumerate(loo.split(X)):
        print(f"Train progress: {i / len(y)*100:.2f}")
        model = RandomForestClassifier(
            n_estimators=75,
            max_depth=12,
            max_features=None,
            min_samples_leaf=2,
            min_samples_split=9,
            class_weight='balanced',
            n_jobs=4,
        )

        model = train_pdl_classifier(X[train_index],y[train_index], generate_all = True)
        pred, probs, unc = predict_pairwise(model, X[train_index], X[test_index])
        loo_probs_n2.append(probs)
        pred, probs, unc = predict_symmetric_pairwise(model, X[train_index], X[test_index])
        loo_probs_n2_sym.append(probs)
        print("Finished predictions.")
    for i, (train_index, test_index) in enumerate(loo.split(X)):
        print(f"Train progress: {i / len(y)*100:.2f}")
        model = RandomForestClassifier(
            n_estimators=75,
            max_depth=12,
            max_features=None,
            min_samples_leaf=2,
            min_samples_split=9,
            class_weight='balanced',
            n_jobs=4,
        )
        model = PairwiseDifferenceClassifier(model)
        model.fit(X[train_index], y[train_index])
        X_screen = np.array(X[test_index])
        y_pred = model.predict_proba(X_screen)

        loo_probs_full.append(y_pred[:,1])
        print("Finished predictions.")
    
    results_to_save = {
    "loo_probs_ncr": np.vstack(loo_probs),                      # from predict_pairwise (ncr, no sym)
    "loo_probs_ncr_sym": np.vstack(loo_probs_sym),              # from predict_symmetric_pairwise (ncr)
    "loo_probs_n2": np.vstack(loo_probs_n2),                    # from predict_pairwise with generate_all=True
    "loo_probs_n2_sym": np.vstack(loo_probs_n2_sym),            # from predict_symmetric_pairwise with generate_all=True
    "loo_probs_full": np.vstack(loo_probs_full),                # from PairwiseDifferenceClassifier
    "y_true": y                                                 # add true labels for reference
    }
    np.savez(args.output, **results_to_save)
    print(f"Saved results to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pairwise difference learning classifier")
    parser.add_argument('--X_train', type=str, required=True, help='Path to input train data')
    parser.add_argument('--y', type=str, required=True, help='Path to input labels data')
    parser.add_argument('--output', type=str, default="output.npz", help='Path to output file')


    args = parser.parse_args()
    main(args)
