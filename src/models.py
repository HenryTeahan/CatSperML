from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from rdkit.Chem import Draw
import joblib
import os


def get_feature_property(feature_index):
    """ Maps feature index to a property (logP, HBA, HBD, Noatom). """
    feature_map = {0: 0, 1: 1, 2: 2, 3: 3}  # Maps feature mod 3 to correct property
    centroid_index = feature_index // 4  # Correct centroid index
    
    return feature_map.get(feature_index % 4, "Unknown"), centroid_index


def get_feature_name(feature_index):
    """Returns the actual feature meaning (logP, HBA, HBD) for a given feature index."""
    feature_map = {0: "logP", 1: "HBA", 2: "HBD", 3: "Atom"}  # Maps feature type
    property_type, centroid_index = get_feature_property(feature_index)
    
    feature_name = feature_map.get(property_type, "Unknown")  # Gets property name
    return f"{feature_name} (Centroid {centroid_index})"

def train(X_train, IC50):

    parent_dir = os.pardir

    os.makedirs(os.path.join(parent_dir, "results/models"), exist_ok=True)
    os.makedirs(os.path.join(parent_dir, "results/figures"), exist_ok=True)
    #os.makedirs(os.path.join(parent_dir, "result/"), exist_ok=True)

    feature_names = [f"Feature {get_feature_name(i)}" for i in range(X_train.shape[1])]
    final_tree = DecisionTreeClassifier(max_depth=None, random_state=0)
    final_tree.fit(X_train, IC50)
    
    plt.figure(figsize=(20, 20))
    fig = plot_tree(
        final_tree,
        filled=True,
        feature_names=feature_names, # Add correspondig feature class
        rounded=True,
        fontsize=8
    )

    save_mod = os.path.join(parent_dir, "results/models", "decision_tree.pkl")
    save_pic = os.path.join(parent_dir, "results/figures", "decision_tree.png")

    joblib.dump(final_tree, save_mod)
    plt.savefig(save_pic, dpi=400, bbox_inches="tight") 

    y_pred = final_tree.predict(X_train)
    
    if np.sum(y_pred - IC50) == 0:
        print("Training converged")
    else:
        print("Training did not converge! Consider changing the descriptor metrics!")
    return final_tree

def train_and_predict(mols_screen, mols_train, IC50, X_train, X_screen):
  #  all_mols = np.hstack([mols_screen, mols_train]).flatten()
    feature_names = [f"Feature {get_feature_name(i)}" for i in range(X_train.shape[1])]
    final_tree = DecisionTreeClassifier(max_depth=None, random_state=0)
    final_tree.fit(X_train, IC50)
    
    plt.figure(figsize=(20, 20))
    plot_tree(
        final_tree,
        filled=True,
        feature_names=feature_names, # Add correspondig feature class
        rounded=True,
        fontsize=8
    )
    plt.title("Aggregated Decision Tree")
    plt.show()
    y_pred = final_tree.predict(X_train)
    cmt = confusion_matrix(IC50, y_pred)
    cmt
    preds = final_tree.predict(X_screen)
    print(f'shape of descriptor: {np.shape(X_train)[1]}')

    return preds, mols_screen, X_train, X_screen, final_tree, y_pred

def get_split_order(final_tree):
    # Extracts features by branch  true first and then false
    tree = final_tree.tree_
    split_features = tree.feature 
    split_thresholds = tree.threshold
    k=0
    split_info = []
    seen_features = set()
    for node_id, feature in enumerate(split_features):
        if feature != -2 and feature not in seen_features:
            seen_features.add(feature)
            split_info.append((feature, split_thresholds[node_id]))  
            k+=1
            print(f'split number = {k}, feature: {feature}, criteria: (<=) {split_thresholds[node_id]}')

    return split_info

def extract_features(final_tree, depth):
    ## Extracts features by importance
    sorted_importance = []
    sorted_importance_idx = []
    d = 0
    for idx in np.argsort(-final_tree.feature_importances_):
        if d < depth:
            sorted_importance.append(final_tree.feature_importances_[idx])
            sorted_importance_idx.append(idx)
            print(f'feature = {idx}, importance: {final_tree.feature_importances_[idx]}')
        d+=1
    return sorted_importance, sorted_importance_idx

def get_test_path(final_tree, test_object, feature_names=None, return_prop=False): #TODO: remove featurenames from implementation (outdated)
    ''' 
    final_tree : final trained decision tree
    test_object : X_screen[index]
    feature_names : 
    '''
   # print(test_object)
    tree = final_tree.tree_
    
    node_id = 0  # Start at the root
    path_info = []
    propd  = []
    while tree.feature[node_id] != -2:  # While not a leaf
        feature = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        feature_value = test_object[feature]
        
        direction = "left" if feature_value <= threshold else "right"
        path_info.append((feature, threshold, direction))
        property_type, centroid_index = get_feature_property(int(feature))
        
        
   #   print(f"Node {node_id}: Feature {feature}, Property_type: {property_type} ({feature_names[feature] if feature_names else 'F'+str(feature)}) "
   #         f"<= {threshold} ? {feature_value} -> Go {direction}")
      #  print(f'centroids index {centroid_index}, feature = {feature}, property = {property_type}')
        if feature_value <= threshold:
            node_id = tree.children_left[node_id]
        else:
            node_id = tree.children_right[node_id]
        if return_prop:
            propd.append(property_type)
   # print(f"Reached leaf node {node_id}")
    if return_prop:
        return path_info, propd
    else:
        return path_info
    
def trace_path(final_tree, X_train, X_screen, indices, mols_train, hit_number, alt_true_IC50, true_id = None, true_index = False, return_all = False):
    
    path_info = get_test_path(final_tree, X_screen[indices[hit_number]].reshape(-1,1))
    if true_index:
        path_info = get_test_path(final_tree, X_screen[true_id].reshape(-1,1))
    shared_path1 = []
    shared_path0 = []
    decision_paths = final_tree.decision_path(X_train).toarray()
    feature_id_1 = path_info[-1][0]
    splitting_nodes_1 = np.where(final_tree.tree_.feature == feature_id_1)[0]
    if len(splitting_nodes_1) == 0:
        print(f"Feature {feature_id_1} not found in decision tree splits.")
        return [], None 
    print(f"Nodes splitting on Feature {feature_id_1}: {splitting_nodes_1}")

    samples_splitting_at_feature_1 = np.where(decision_paths[:, splitting_nodes_1].sum(axis=1) > 0)[0]
    print(f"Samples splitting at Feature {feature_id_1}: {samples_splitting_at_feature_1[:20]}")
    shared_path1.extend(samples_splitting_at_feature_1)
    feature_id_0 = path_info[-2][0]
    splitting_nodes_0 = np.where(final_tree.tree_.feature == feature_id_0)[0]

    print(f"Nodes splitting on Feature {feature_id_0}: {splitting_nodes_0}")

    samples_splitting_at_feature_0 = np.where(decision_paths[:, splitting_nodes_0].sum(axis=1) > 0)[0]
    print(f"Samples splitting at Feature {feature_id_0}: {samples_splitting_at_feature_0[:20]}")
    shared_path0.extend(samples_splitting_at_feature_0)

    common_samples = list(set(shared_path1) & set(shared_path0))
    print(f"Common samples (indices in X_train): {common_samples}")
    
    molecules_splitting_at_feature = [mols_train[idx] for idx in common_samples]
    ic50_mols_splitting_at_feature = [alt_true_IC50[idx] for idx in common_samples]
    print(f"Found {len(molecules_splitting_at_feature)} molecules following the common prior path.")
    selected_mols = [molecules_splitting_at_feature[idx] for idx in range(len(molecules_splitting_at_feature))]
    legends = [f'IC50: {ic50_mols_splitting_at_feature[idx]}' for idx in range(len(selected_mols))]
    img = Draw.MolsToGridImage(selected_mols, molsPerRow=4, subImgSize=(600, 600), maxMols=25, legends = legends) 
    if return_all:
        return molecules_splitting_at_feature, img, common_samples, X_screen[indices[hit_number]].reshape(-1,1), hit_number
    return molecules_splitting_at_feature, img

#def get_distances(final_tree, test_object, molecule, descriptors, centroids, numb_atom=1):
#
#    path_info = get_test_path(final_tree, test_object)
#    temp_dis = []
#    distances_from_centroids = {}
#    for feature, threshold, direction in path_info:
#        property_type, centroid_index = get_feature_property(int(feature))
#
#        if centroid_index >= len(centroids):
#            continue  
#
#        centroid = centroids[centroid_index]
#        coords = get_coords(molecule)
#        distances = np.linalg.norm(centroid - coords, axis=1)
#        closest_atoms = np.argsort(distances)[:numb_atom]
#
#        distances_from_centroids[feature] = distances[closest_atoms].tolist()
#        temp_dis.append(distances[closest_atoms].tolist())
#    me_dis = np.mean(temp_dis)
#    #me_dis = np.mean(distances_from_centroids[feature] for feature, _,_ in path_info)
#    return distances_from_centroids, me_dis

def retrieve_molecules_from_node(decision_tree, X_train, X_screen, mols_train, mols_screen, node_id, IC50, alt_true_IC50
                                ):
    """
    Retrieves molecules from a specific node of the tree.
    """
    decision_paths_train = decision_tree.decision_path(X_train).toarray()
    decision_paths_screen = decision_tree.decision_path(X_screen).toarray()

    train_indices = np.where(decision_paths_train[:,   node_id] > 0)[0]
    screen_indices = np.where(decision_paths_screen[:, node_id] > 0)[0]
    #print('train:',train_indices)
    #print('test', screen_indices)
  #  print(screen_indices, node_id)
    mols_train_node = [mols_train[i] for i in train_indices]
    class_train_node = [IC50[i] for i in train_indices]
    acc = [float(alt_true_IC50[i]) for i in train_indices]

    mols_screen_node = [mols_screen[i] for i in screen_indices]
    X_train_node = [X_train[i] for i in train_indices]
    X_screen_node = [X_screen[i] for i in screen_indices]

    return mols_train_node, mols_screen_node, X_train_node, X_screen_node, class_train_node, acc, screen_indices, train_indices

import time