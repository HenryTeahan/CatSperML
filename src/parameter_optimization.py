from descriptors import make_descriptors_fast
from evaluation import custom_metric
from sklearn.model_selection import LeaveOneOut
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import matthews_corrcoef, confusion_matrix

def leave_one_out_cv(mols_train, IC50, X, rs = 0):
    loo = LeaveOneOut()
    true_IC50 = np.zeros(len(IC50))
    pred_IC50 = np.zeros(len(IC50))
    mols_list = []
    trees_ = []
    metr = []

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = IC50[train_index], IC50[test_index]
        mol_test = mols_train[test_index[0]]
        
        final_tree = DecisionTreeClassifier(max_depth=None, random_state=rs)
        final_tree.fit(X_train, y_train)
        trees_.append(final_tree)
        y_pred = final_tree.predict(X_test)[0]
        true_IC50[test_index] = y_test[0]; pred_IC50[test_index] = y_pred; mols_list.append(mol_test)
        #metric = custom_metric(final_tree, centroids, X_test, mol_test, eps)
        #metr.append(metric)
    return true_IC50, pred_IC50,  mols_list, trees_#, metr

def evaluate_combination(mols_train, mols_screen, IC50, params, idx, tree_votes = False, num_rand_states = 100):
    #del X_train, X_screen, centroids, mols_list, trees_
    #gc.collect()  # Collect garbage explicitly
    eps, n_clusters, allow_dis, mode = params
    #print(f"Progress: {int((idx / total) * 100)}%")
    
    X_train, X_screen, centroids = make_descriptors_fast(
        mols_train, mols_screen,
        n_clusters=n_clusters, eps=eps,
        allow_dis=allow_dis, DBScan_mode=True,
        min_samples=n_clusters, distance_based=True,
        return_centroids=True
    )
    print('Descriptors generated')
    ### New code starts here ------------
    if tree_votes == True:
        mcc_scores = []
        eps_vals = []

        for r in range(num_rand_states):
            true_IC50, pred_IC50, mols_list, trees_, metr = leave_one_out_cv(mols_train, 
                                                                             IC50, X_train, centroids, eps, r)

            mcc = matthews_corrcoef(true_IC50, pred_IC50)

            mcc_scores.append(mcc)
            eps_vals.append(eps)

        return {
            #'idx': idx,
            'eps': eps,
            #'n_clusters': n_clusters,
            #'allow_dis': allow_dis,
            #'mode': mode,
            #'true_IC50': true_IC50,
            #'pred_IC50': pred_IC50,
            #'molss_list': mols_list,
            #'trees': trees_,
            #'metr': metr,
            'mcc': mcc_scores,
            #'cm': cm
        }
    ### New code ends here ------------
    else:
        true_IC50, pred_IC50, mols_list, trees_, metr = leave_one_out_cv(mols_train, 
                                                                         IC50, X_train, centroids, eps, rs=0)
    #print('hi')
    mcc = matthews_corrcoef(true_IC50, pred_IC50)
    cm = confusion_matrix(true_IC50, pred_IC50)
    #print(mcc)
    #del X_train, X_screen, centroids, mols_list, trees_
    #gc.collect()  # Collect garbage explicitly
    
    return {
        'idx': idx,
        'eps': eps,
        'n_clusters': n_clusters,
        'allow_dis': allow_dis,
        'mode': mode,
        'true_IC50': true_IC50,
        'pred_IC50': pred_IC50,
        'molss_list': mols_list,
        'trees': trees_,
        'metr': metr,
        'mcc': mcc,
        'cm': cm
    }