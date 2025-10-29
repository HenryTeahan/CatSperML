from curses.ascii import alt
from multiprocessing import Value
#from tkinter import N
from tracemalloc import start
from scipy import cluster
from sklearn.tree import DecisionTreeClassifier, _tree, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import DBSCAN
from sklearn.model_selection import LeaveOneOut, train_test_split, cross_val_score
from joblib import Parallel, delayed
from itertools import combinations
from scipy.linalg import norm
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sbn
import networkx as nx
from numba import njit, prange

from sklearn.cluster import AgglomerativeClustering
import re
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import (AllChem, Draw, Descriptors, rdDistGeom, rdMolTransforms,
                         rdMolDescriptors, rdFMCS)
import pygraphviz
from rdkit.Chem.Draw import (IPythonConsole, MolsToGridImage, MolDraw2DSVG)
from rdkit.Geometry import Point2D
from PIL import Image
from io import BytesIO
from copy import deepcopy
from IPython.core.display import HTML, display_html, display_svg, SVG
from itertools import product
import os

IPythonConsole.drawOptions.addAtomIndices = True
IPythonConsole.molSize = (500, 500)

def get_coords(m):
  X = []
  for i, atom in enumerate(m.GetAtoms()):
    positions = m.GetConformer(0).GetAtomPosition(i)
    X.append([positions.x, positions.y])

  return np.array(X) # Gets coordinates of all non-H points 

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

# 3-feature code below
#def get_feature_property(feature_index):
#    """ Maps feature index to property (logP, HBA, HBD). """
#    feature_map = {0: "logP", 1: "HBA", 2: "HBD"}
#   # print(feature_index % 3)
#    property_type = feature_map[feature_index % 3]  # Correct descriptor type
#    centroid_index = feature_index // 3  # Correct centroid index
#    return property_type, centroid_index

#def get_feature_property(feature_index):
#    """ Maps feature index to a property (logP, HBA, HBD). """
#    feature_map = {0: 0, 1: 1, 2: 2}  # Maps feature mod 3 to correct property
#    centroid_index = feature_index // 3  # Correct centroid index
#    
#    return feature_map.get(feature_index % 3, "Unknown"), centroid_index

# With noatom:
def get_feature_property(feature_index):
    """ Maps feature index to a property (logP, HBA, HBD, Noatom). """
    feature_map = {0: 0, 1: 1, 2: 2, 3: 3}  # Maps feature mod 3 to correct property
    centroid_index = feature_index // 4  # Correct centroid index
    
    return feature_map.get(feature_index % 4, "Unknown"), centroid_index

#
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
    
def get_logP(m): #per atom hydrophobicity - if atom is hydrogen, finds contributing heavy atom and assigns logP
  mol = Chem.Mol(m)
  mol = Chem.AddHs(mol)
  contribs = rdMolDescriptors._CalcCrippenContribs(mol)
  logP = [x for x,y in contribs]
  for i,atom in enumerate(mol.GetAtoms()):
    if atom.GetAtomicNum() == 1:
      idx = atom.GetNeighbors()[0].GetIdx()
      logP[idx] += logP[i]

    return logP[:mol.GetNumHeavyAtoms()]

def get_HBs(m):
  NumHBD = Chem.MolFromSmarts("[N&!H0&v3,N&!H0&+1&v4,O&H1&+0,S&H1&+0,n&H1&+0]") # Donors (N, N+, O, S, aromat. N)
  NumHBA = Chem.MolFromSmarts("[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$(" # Acceptors (O or S in hydroxyl groups, O or S w.o. hydrogens or with a neg. charge)
                "[N;v3;!$(N-*=!@[O,N,P,S])]),$([nH0,o,s;+0])]") #Nitrogen w. valence 3, aromatic N etc.

  HBDs = np.array(m.GetSubstructMatches(NumHBD)).flatten()
  HBAs = np.array(m.GetSubstructMatches(NumHBA)).flatten()

  return HBDs, HBAs
def get_logP(m):
    mol = Chem.Mol(m)
    #mol = Chem.RemoveAllHs(mol)
    mol = Chem.AddHs(mol)
    
    contribs = rdMolDescriptors._CalcCrippenContribs(mol)
    
    logP = [x for x, y in contribs]  # Extract logP contributions

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:  # Hydrogen
            heavy_idx = atom.GetNeighbors()[0].GetIdx()
            logP[heavy_idx] += logP[atom.GetIdx()]  # Assign Hydrogen's contribution
    
    return logP[:mol.GetNumHeavyAtoms()]  # Ensure only heavy atoms are returned


    
def get_HBs(m):
    NumHBD = Chem.MolFromSmarts("[N&!H0&v3,N&!H0&+1&v4,O&H1&+0,S&H1&+0,n&H1&+0]") # Donors (N, N+, O, S, aromat. N)
    NumHBA = Chem.MolFromSmarts(
        "[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),"
        "$([O,S;H0;v2]),"
        "$([O,S;-]),"
        "$([N;v3;!$(N-*=!@[O,N,P,S])]),"
        "$([nH0,o,s;+0])]"
    )  
    HBDs = np.array(m.GetSubstructMatches(NumHBD)).flatten()
    HBAs = np.array(m.GetSubstructMatches(NumHBA)).flatten()

    return HBDs, HBAs

def uncharge(mol):
  pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
  at_matches = mol.GetSubstructMatches(pattern)
  at_matches_list = [y[0] for y in at_matches]
  if len(at_matches_list) > 0:
    for at_idx in at_matches_list:
      atom = mol.GetAtomWithIdx(at_idx)
      chg = atom.GetFormalCharge()
      hcount = atom.GetTotalNumHs()
      atom.SetFormalCharge(0)
      atom.SetNumExplicitHs(hcount - chg)
      atom.UpdatePropertyCache()
  return mol

def get_feature_name(feature_index):
    """Returns the actual feature meaning (logP, HBA, HBD) for a given feature index."""
    feature_map = {0: "logP", 1: "HBA", 2: "HBD", 3: "Atom"}  # Maps feature type
    property_type, centroid_index = get_feature_property(feature_index)
    
    feature_name = feature_map.get(property_type, "Unknown")  # Gets property name
    return f"{feature_name} (Centroid {centroid_index})"

def visualize_molecule_decision_path(final_tree, test_object, molecule, descriptors, centroids, feature_names=None, numb_atom=1):
    path_info = get_test_path(final_tree, test_object, feature_names)
    svgs = []

    def edit_molecule(m, centroid_position):
        """ Adds a Neon (Ne) atom at the specified centroid position. """
        rwMol = Chem.RWMol(m)
        a = Chem.Atom(10)  # Neon atom
        centroid_idx = rwMol.AddAtom(a)

        if rwMol.GetNumConformers() == 0:
            rwMol.AddConformer(Chem.Conformer(rwMol.GetNumAtoms()))
        rwMol.GetConformer().SetAtomPosition(centroid_idx, (centroid_position[0], centroid_position[1], 0.0))

        return rwMol.GetMol(), centroid_idx

    for step, (feature, threshold, direction) in enumerate(path_info):
        property_type, centroid_index = get_feature_property(feature)
        centroid_index = int(centroid_index)
        print(f'centroid_index: {centroid_index}, property_type: {property_type}')

        descriptor_value = float(test_object[feature]) 
        centroid = centroids[centroid_index]  
        coords = get_coords(molecule)  
        distances = np.linalg.norm(centroid - coords, axis=1)
        closest_atoms = np.argsort(distances)[:numb_atom].tolist()

        colors = {}
        temp_molecule = Chem.Mol(molecule)
        new_atom_idx = None  
       # if (descriptors[centroid_index, 3] == 0 and distances.min()>eps):  # We assign Neon if no atom is a req. or further than eps
        if distances.min() > eps:
            temp_molecule, new_atom_idx = edit_molecule(temp_molecule, centroid)  
          #  if new_atom_idx is not None:
            colors[new_atom_idx] = (0, 1, 1, 0.7)  
        
        else:# closest_atoms:  # If there are atoms within eps
            for atom_idx in closest_atoms:
                atom_idx = int(atom_idx)
                if descriptor_value <= threshold:
                    colors[atom_idx] = (0, 1, 0, 0.7)  
                else:
                    colors[atom_idx] = (1, 0, 0, 0.6)  
                   # print('Distance =', distances.min(), eps)
        
        

        highlight_atoms = [idx for idx in colors.keys() if idx < temp_molecule.GetNumAtoms()]
        valid_colors = {idx: colors[idx] for idx in highlight_atoms}

        drawer = rdMolDraw2D.MolDraw2DSVG(400, 400)
        drawer.DrawMolecule(temp_molecule, highlightAtoms=highlight_atoms, highlightAtomColors=valid_colors)
        drawer.FinishDrawing()
        raw_svg = drawer.GetDrawingText().replace('svg:', '')
        
       # if (descriptors[centroid_index, 3] == 0 or distances.min() > eps):
        print(np.shape(descriptors), centroid_index)
        if (descriptors[centroid_index, 3] == 0 or distances.min() > eps):
            if (distances.min() > eps or descriptors[centroid_index, 3]==1):
                feature_value_text = f"No atom! Defaulting to {get_feature_name(feature)}=0"
            elif get_feature_name(feature) == 'Atom':
                feature_value_text = f""
            else:
                feature_value_text = f"Atom exists - {get_feature_name(feature)} = {descriptor_value:.3f}"

            #feature_value_text = (
            #    f"No Atom! Defaulting to 0"
            #    if (distances.min() > eps or descriptors[centroid_index, 3] == 1)
            #    elif f" " if get_feature_name(feature) == 'Atom'
            #    else f"Atom exists --- {get_feature_name(feature)} = {descriptor_value:.4f}"
            
        else:
            feature_value_text = f"Descriptor value {descriptor_value:.3f}"
        #feature_value_text = f"desc = {descriptor_value:.5f}, dist = {distances.min()}"

        decision_text_lines = [
            f"Step {step+1}: Feature {feature} ({feature_names[feature] if feature_names else 'F' + str(feature)})",
            f"Feature meaning: {get_feature_name(feature)} ≤ {threshold:.3f}",
            f"{feature_value_text} <--> Go {direction}"
        ]

        text_annotations = '''
            <text x="10" y="25" font-size="14" fill="black">
                {}
            </text>
        '''.format(''.join(f'<tspan x="0" dy="{i * 20}">{line}</tspan>' for i, line in enumerate(decision_text_lines)))

        raw_svg = re.sub(r'(<\/svg>)', text_annotations + r'<rect width="100%" height="200" fill="transparent" />' +f'\n\n\n\n\n'+ r'\1', raw_svg)
        raw_svg = re.sub(r'(<svg [^>]+height="[^"]+")', r'\1 200', raw_svg)

        svgs.append(raw_svg)

    cols_per_row = 3
    num_svgs = len(svgs)
    num_rows = (num_svgs + cols_per_row - 1) // cols_per_row

    grid_width = cols_per_row * 600
    grid_height = num_rows * 600
    combined_svg = f"""
    <svg width="{grid_width}" height="{grid_height}" xmlns="http://www.w3.org/2000/svg">
        {''.join(f'<g transform="translate({(i % cols_per_row) * 600}, {(i // cols_per_row) * 400})">{svg}</g>' for i, svg in enumerate(svgs))}
    </svg>
    """

    display_svg(combined_svg, raw=True)

def get_distances(final_tree, test_object, molecule, descriptors, centroids, feature_names=None, numb_atom=1, verbose = False):
    path_info = get_test_path(final_tree, test_object, feature_names)
    confidence_scores = {}  
    distances_from_centroids = {}  

    for step, (feature, threshold, direction) in enumerate(path_info):
        property_type, centroid_index = get_feature_property(feature)

        descriptor_value = test_object[feature]
        centroid = centroids[centroid_index]
        coords = get_coords(molecule) 
        distances = np.linalg.norm(centroid - coords, axis=1)  
        closest_atoms = np.argsort(distances)[:numb_atom].tolist()

        distances_from_centroids[feature] = [distances[atom_idx] for atom_idx in closest_atoms]

        avg_distance = np.mean(distances_from_centroids[feature])
        confidence_scores[feature] = 1 / (1 + avg_distance)  

    confidence_values = np.array(list(confidence_scores.values())) if confidence_scores else np.array([0])
    distance_values = np.array([d for feature_distances in distances_from_centroids.values() for d in feature_distances]) if distances_from_centroids else np.array([0])
    mean_confidence = np.mean(confidence_values)  
    if len(distance_values) > 0 and np.sum(distance_values) > 0: 
        inv_distances = 1 / (1 + distance_values)
        weighted_confidence = np.average(confidence_values, weights=inv_distances)
    else:
        weighted_confidence = mean_confidence
    if verbose:
        print(f"Mean Confidence: {mean_confidence:.4f}")
        print(f"Weighted Confidence: {weighted_confidence:.4f}")

    return distances_from_centroids, confidence_scores, weighted_confidence
def make_descriptors(mols_train, mols_screen, n_clusters, eps, allow_dis, DBScan_mode = False, min_samples=None, distance_based=False, return_centroids = False, mode = 'linear'): 
    """Descriptor generation function.
    Mols_train: Training rdkit.mol objects
    Mols_screen: Screening rdkit.mol objects
    n_clusters: Parameter for Agglomerative Clustering
    eps: Clustering radius for both Agg.Clustering and DBscan
    allow_dis: Distance threshold for descriptor calculation. IF ONLY NEAREST MOLECULE DESIRED, SET ALLOW_DIS = EPS
    """
    start_time = time.time()
    all_mols = mols_train
    X = []

    for m in all_mols:
        for i, atom in enumerate(m.GetAtoms()):
            positions = m.GetConformer(0).GetAtomPosition(i)
            X.append([positions.x, positions.y])

    X = np.array(X)

    def DBScan(X, n_clusters, eps=eps):
        if DBScan_mode == False:
            agglom = AgglomerativeClustering(n_clusters=n_clusters)
            labels = agglom.fit_predict(X)
        if DBScan_mode == True:
            print('DBscan clustering')
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
        unique_labels = set(labels)
        clusters = {label: X[labels == label] for label in unique_labels if label != -1}
        centroids = {label: cluster.mean(axis=0) for label, cluster in clusters.items()}
        centroids = np.stack(list(centroids.values()))
        return centroids

    centroids = DBScan(X, n_clusters)
    if distance_based:
        print(eps, allow_dis)
        def compute_descriptors(mols, centroids, eps,allow_dis):
            descriptors = []

            for mm, m in enumerate(mols):
                
                descriptor = np.full((len(centroids), 4), [0,0,0,0], dtype=float) 
                m = Chem.RemoveAllHs(m)

                coords = get_coords(m) 
                logP = get_logP(m)  
                #print('oldlogp',logP)
                HBDs, HBAs = get_HBs(m)
                if len(logP) != len(m.GetAtoms()):
                    print(f"ERROR: logP has a mismatch in length: {mm},{m}, {len(logP)} vs. {len(m.GetAtoms())} atoms.")
                    break  
                for i, atom in enumerate(m.GetAtoms()):
                    if atom.GetAtomicNum() == 1:
                        continue
                    distances = np.linalg.norm(centroids - coords[i], axis=1)
                    within_eps = np.where(distances <= allow_dis)[0]

                    if len(within_eps) > 0:  
                        
                        for k, cluster_idx in enumerate(within_eps):
                            if mode == 'linear':
                                
                                d = distances[cluster_idx]

                                if np.isnan(logP[i]) or np.isinf(logP[i]):
                                    print(f"ERROR: Invalid logP value: {logP[i]} for atom {i}, cluster_idx {cluster_idx}.")
                                    break
                                #print(f"Atom {i}: distance = {d}, weight = {1/eps/d if d>eps else 1}")
                                descriptor[cluster_idx][0] += (logP[i]/(d/eps) if d>eps else logP[i])
                               # print('cumlogPold', descriptor[cluster_idx][0])
                                descriptor[cluster_idx][1] += (int(i in HBAs)/(d/eps) if d>eps else int(i in HBAs))
                                descriptor[cluster_idx][2] += (int(i in HBDs)/(d/eps) if d>eps else int(i in HBDs))
                                descriptor[cluster_idx][3] = 1
                
                            elif mode == 'square':
                                d = distances[cluster_idx]
                                descriptor[cluster_idx][0] += (logP[i]/(d/eps)**2 if d>eps else logP[i])
                                descriptor[cluster_idx][1] += (int(i in HBAs)/(d/eps)**2 if d>eps else int(i in HBAs))
                                descriptor[cluster_idx][2] += (int(i in HBDs)/(d/eps)**2 if d>eps else int(i in HBDs))
                                descriptor[cluster_idx][3] = 1    
                descriptors.append(np.round(descriptor.flatten(), decimals=3))

            return np.array(descriptors)
    else:
        def compute_descriptors(mols, centroids, eps, allow_dis = None):
                descriptors = []
            
                for m in mols:
                    # Initialize descriptor with 4 features per centroid
                    descriptor = np.full((len(centroids), 4), [0,0,0,0], dtype=float)  # Adding presence flag
            
                    coords = get_coords(m) 
                    logP = get_logP(m)  
                    HBDs, HBAs = get_HBs(m)
            
                    for i, atom in enumerate(m.GetAtoms()):
                        distances = np.linalg.norm(centroids - coords[i], axis=1)
                        within_eps = np.where(distances <= eps)[0]
            
                        if len(within_eps) > 0:  # If atom is within epsilon distance of a centroid
                            for cluster_idx in within_eps:
                                descriptor[cluster_idx][0] += logP[i]   # Adjust logP
                                descriptor[cluster_idx][1] += int(i in HBAs)   # Count HBAs
                                descriptor[cluster_idx][2] += int(i in HBDs)   # Count HBDs
                                descriptor[cluster_idx][3] = 1.0  # Mark presence of an atom
            
                    descriptors.append(descriptor.flatten())  
                    
                return np.array(descriptors)

    X_train = compute_descriptors(mols_train, centroids, eps, allow_dis)
    X_screen = compute_descriptors(mols_screen, centroids, eps, allow_dis)
    end_time = time.time()
    print(f"Made {len(mols_screen) + len(mols_train)} descriptors in {end_time - start_time} seconds {(len(mols_screen) + len(mols_train))*len(X_train[0])*64/(end_time - start_time):.2f} bits / s")    

    if return_centroids == True:
        return X_train, X_screen, centroids
    return X_train, X_screen

def draw_centroids(mols_train, n_clusters, min_samples,eps, DBScan = False, draw = False, return_dat = False):
    X = []
    for m in mols_train:
        for i, atom in enumerate(m.GetAtoms()):
            positions = m.GetConformer(0).GetAtomPosition(i)
            X.append([positions.x, positions.y])
    X = np.array(X)
    agglom = AgglomerativeClustering(n_clusters=n_clusters, linkage = 'ward')
    labels = agglom.fit_predict(X)
    if DBScan:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
    #plt.figure(figsize=(10, 8))
    
    unique_labels = set(labels)
    
    clusters = {label: X[labels == label] for label in unique_labels if label != -1}
    centroids = {label: cluster.mean(axis=0) for label, cluster in clusters.items()}
    centroids_array = np.stack(list(centroids.values()))
    num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  

    cmap = plt.get_cmap('viridis')  # This will work without warning now.
    colors = {label: cmap(i / num_clusters) for i, label in enumerate(unique_labels) if label != -1}
    marker_styles = itertools.cycle(['o', 's', 'D', 'P', '*', 'X', '^', '<', '>'])
    markers = {label: next(marker_styles) for label in unique_labels if label != -1} 
    if return_dat:
        return clusters, centroids, colors, markers
    #linkage_matrix = shc.linkage(X, method='ward') 
##  
    if draw:
        plt.figure(figsize=(10, 7))
        #shc.dendrogram(linkage_matrix)  
        #plt.title("Agglomerate clustering tree")
        #plt.xlabel("Training molecuels")
        #plt.ylabel("Ward's distance")
        #plt.show()

        for label, cluster in clusters.items():
            plt.scatter(cluster[:, 0], cluster[:, 1], color=colors[label], marker=markers[label])#, label=f"Cluster {label}")
        for label, centroid in centroids.items():
            plt.scatter(centroid[0], centroid[1], color="black", marker="X", s=150)#, label=f"Centroid {label}")
        #for label, centroid in centroids.items():
        #    circle = plt.Circle((centroid[0], centroid[1]), radius=eps, color='black', fill=False, linewidth=2)
        #    plt.gca().add_patch(circle)
#   
        if DBScan:
            plt.title(f'DBSCAN Clustering (min_samples = {min_samples}, eps = {eps})')
        else:
            plt.title(f'Agglomerate Clustering (n_clusters = {n_clusters}, eps = {eps})')
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")

        plt.show()
    centroids = np.stack(list(centroids.values()))

    return centroids

#def visualize(final_tree, X, mol, centroids, eps, allow_dis, textanot = False, tex = None,draw = False, 
#              verbose = False, font = 20, size=(800, 800), baseFont = 0.4, tex_pos = Point2D(0,5), textsize = 30):
#    if draw:
#        size = size
#        d2d = Draw.MolDraw2DCairo(*size)
#        dopts = d2d.drawOptions()
#        dopts.useBWAtomPalette()
#        dopts.baseFontSize = baseFont
#        dopts.padding = 0.2 # default is 0.05
#        d2d.DrawMolecule(mol)
#        if textanot:
#            text = tex
#            text_position = tex_pos  # Adjust coordinates as needed
#            d2d.SetFontSize(textsize)
#            d2d.DrawString(text, text_position)
#    test_path = get_test_path(final_tree, X)
#
#
#    def draw_rectangle(center, d2d, color=(1,0,0,1), size=0.6):
#        scale=size
#        x,y = center
#        x,y = float(x), float(y)
#
#        d2d.SetLineWidth(0)
#        pos1 = Point2D(x-scale, y+scale)
#        pos2 = Point2D(x+scale, y-scale)
#        d2d.SetColour(color)
#        d2d.DrawRect(pos1, pos2)
#
#    def draw_X(center, d2d, color=(0,0,0,1.0), size=0.6):
#        x,y = center
#        x,y = float(x), float(y)
#
#        scale = size/2
#        d2d.SetColour(color)
#        d2d.SetLineWidth(4)
#
#        pos1 = Point2D(x-scale,y+scale)
#        pos2 = Point2D(x+scale,y-scale)
#        d2d.DrawLine(pos1, pos2)
#        pos3 = Point2D(x-scale,y-scale)
#        pos4 = Point2D(x+scale,y+scale)
#        d2d.DrawLine(pos3, pos4)
#
#    def draw_HA(center, d2d, isThere=True):
#        d2d.SetFontSize(25)
#        x,y = center
#        x,y = float(x), float(y)
#        draw_rectangle(center, d2d, color=(0.2, 0, 0.6, 0.4))
#        d2d.SetDrawOptions(d2d.drawOptions())
#        d2d.SetColour((0,0,0))
#        d2d.DrawString("H", Point2D(x-2.3*0.1275,y))
#        d2d.DrawString("A", Point2D(x+2.3*0.1275,y))
#        if not isThere:
#            draw_X(center, d2d, color=(1,0,0,0.4), size=0.5)
#
#    def draw_HD(center, d2d, isThere=True):
#        d2d.SetFontSize(25)
#        x,y = center
#        x,y = float(x), float(y)
#        draw_rectangle(center, d2d, color=(0.9, 0.5, 0.0, 0.5))
#        d2d.SetDrawOptions(d2d.drawOptions())
#        d2d.SetColour((0,0,0))
#        d2d.DrawString("H", Point2D(x-0.1275,y))
#        d2d.DrawString("D", Point2D(x+0.1275,y))
#        if not isThere:
#            draw_X(center, d2d, color=(1,0,0,0.65), size=0.5)
#
#    def draw_checkmark(center, d2d, color=(0, 0.8, 0.1, 1), size=0.6):
#        scale=size/2
#        x,y = center
#        x,y = float(x), float(y)
#
#        d2d.SetColour(color)
#        d2d.SetLineWidth(4)
#        pos1 = Point2D(x-scale,y)
#        pos2 = Point2D(x-(5/30)*scale,y-scale)
#        d2d.DrawLine(pos1, pos2)
#        pos3 = Point2D(x-(7/30)*scale,y-scale)
#        pos4 = Point2D(x+scale,y+scale)
#        d2d.DrawLine(pos3, pos4)
#
#    def draw_logP(center, d2d, threshold, above=True, size=0.8, color=(0.7,0.5,0.3,0.8)):
#        d2d.SetLineWidth(0)
#        x,y=center
#        x,y = float(x), float(y)
#        d2d.SetFontSize(font)
#        #print(center)
#        if above:
#            text = f">{threshold:.1}"
#            color=(0,0.5,0.9,0.5)
#        if not above:
#            text = f"<{threshold:.1}"
#            color=(1,0,0,0.5)
#
#        #print(Point2D(x,y))
#        d2d.SetColour(color)
#        size = float(size) 
#        d2d.DrawArc(Point2D(x,y), size, 0.0, 360.0)
#        d2d.SetDrawOptions(d2d.drawOptions())
#        d2d.SetColour((0,0,0))
#        d2d.DrawString(text, Point2D(x,y))
#
#    epss= []
#    count_under_eps = 0
#    for step in range(len(test_path)):
#        feature, threshold, direction = test_path[step]
#        
#        property_type, centroid_index = get_feature_property(int(feature))
#        feature_value = X[feature]
#        #centroid_index = int(centroid_index)//4 #### NOTE: FEATURE CONNECTED MOD 4 TO CENTROID
#        centroid_value = centroids[centroid_index]
#        coords = get_coords(mol)
#        distances = np.linalg.norm(centroid_value - coords, axis = 1)
#        epss.append(distances.min())
#        closest_atom = np.argmin(distances) #TODO: multiple atoms
#        if distances.min() <= eps:
#            count_under_eps += 1
#
#        feature_class = property_type
#
#        feature_name = get_feature_name(feature_class)
#        if verbose:
#            print(test_path[step])
#            print(distances.min())
#            print(feature_class)
#        if draw:
#            if feature_class == 0:
#                draw_logP(centroid_value, d2d, threshold, feature_value >= threshold)
#
#            elif feature_class == 1:
#                draw_HA(centroid_value, d2d, feature_value >= threshold)
#                #print(direction, 'ha',threshold, feature_value, centroid_index)
#d
#            elif feature_class == 2:
#                draw_HD(centroid_value, d2d, feature_value >= threshold)
#
#            elif feature_class == 3:
#                if distances.min() < allow_dis:
#                    draw_checkmark(centroid_value, d2d)
#                else:
#                    draw_X(centroid_value, d2d)
#            else:
#                print('err',feature_class)
#            d2d.FinishDrawing()
#            d2d.GetDrawingText()
#            bio = BytesIO(d2d.GetDrawingText())
#            img = Image.open(bio)
#    if draw:
#        return img
#    else: 
#        return count_under_eps, epss
    
def visualize(final_tree, X, mol, centroids, eps, allow_dis, textanot = False, tex = None, 
              verbose = False, size=(800, 800), baseFont = 0.5, tex_pos = Point2D(0,5), textsize = 30):
    '''
    final_tree: Trained decision tree model
    X: the descriptor for the molecule that you wish to visualize -> Must have 4 features (columns)
    mol: the rdkit molobject in question
    centroids: The centroids that have been used to generate X
    textanot: True if you wish text annotation
    tex: the text you want to display (textanot must be true to display anything)
    draw: Return the visualization
    verbose: debug statements
    size: Size of the d2d image
    baseFont: Relative size of font
    textsize: Change font size

    -- Returns bytes IO img
    '''
    
    from rdkit.Chem.Draw import rdMolDraw2D

    size = size
    d2d = Draw.MolDraw2DCairo(*size)
    dopts = d2d.drawOptions()
    dopts.useBWAtomPalette()
    dopts.bondLineWidth = 5
    dopts.legendFontSize = 15
    dopts.baseFontSize = baseFont
    dopts.padding = 0.2 # default is 0.05
    d2d.DrawMolecule(mol)
    if textanot:
        text = tex
        text_position = tex_pos  # Adjust coordinates as needed
        d2d.SetFontSize(textsize)
        d2d.DrawString(text, text_position)
    test_path = get_test_path(final_tree, X)


    def draw_rectangle(center, d2d, color=(1,0,0,1), size=0.6):
        scale=size
        x,y = center
        x,y = float(x), float(y)

        d2d.SetLineWidth(0)
        pos1 = Point2D(x-scale, y+scale)
        pos2 = Point2D(x+scale, y-scale)
        d2d.SetColour(color)
        d2d.DrawRect(pos1, pos2)

    def draw_X(center, d2d, color=(0,0,0,1.0), size=1):
        x,y = center
        x,y = float(x), float(y)

        scale = size/2
        d2d.SetColour(color)
        d2d.SetLineWidth(4)

        pos1 = Point2D(x-scale,y+scale)
        pos2 = Point2D(x+scale,y-scale)
        d2d.DrawLine(pos1, pos2)
        pos3 = Point2D(x-scale,y-scale)
        pos4 = Point2D(x+scale,y+scale)
        d2d.DrawLine(pos3, pos4)

    def draw_HA(center, d2d, isThere=True):
        d2d.SetFontSize(25)
        x,y = center
        x,y = float(x), float(y)
        draw_rectangle(center, d2d, color=(0.2, 0, 0.6, 0.4))
        d2d.SetDrawOptions(d2d.drawOptions())
        d2d.SetColour((0,0,0))
        d2d.DrawString("H", Point2D(x-2.3*0.1275,y))
        d2d.DrawString("A", Point2D(x+2.3*0.1275,y))
        if not isThere:
            draw_X(center, d2d, color=(1,0,0,0.4), size=1)

    def draw_HD(center, d2d, isThere=True):
        d2d.SetFontSize(25)
        x,y = center
        x,y = float(x), float(y)
        draw_rectangle(center, d2d, color=(0.9, 0.5, 0.0, 0.5))
        d2d.SetDrawOptions(d2d.drawOptions())
        d2d.SetColour((0,0,0))
        d2d.DrawString("H", Point2D(x-0.1275,y))
        d2d.DrawString("D", Point2D(x+0.1275,y))
        if not isThere:
            draw_X(center, d2d, color=(1,0,0,0.65), size=1)

    def draw_checkmark(center, d2d, color=(0, 0.8, 0.1, 1), size=0.6):
        scale=size/2
        x,y = center
        x,y = float(x), float(y)

        d2d.SetColour(color)
        d2d.SetLineWidth(4)
        pos1 = Point2D(x-scale,y)
        pos2 = Point2D(x-(5/30)*scale,y-scale)
        d2d.DrawLine(pos1, pos2)
        pos3 = Point2D(x-(7/30)*scale,y-scale)
        pos4 = Point2D(x+scale,y+scale)
        d2d.DrawLine(pos3, pos4)

    def draw_logP(center, d2d, threshold, above=True, size=0.8, textsize = textsize, color=(0.7,0.5,0.3,0.8)):
        d2d.SetLineWidth(0)
        x,y=center
        x,y = float(x), float(y)
        if above:
            text = f">{threshold:.2}" # IF greater than :> Red
            #color=(0,0.5,0.9,0.5)
            color=(1,0,0,0.5)
        if not above:
            text = f"<{threshold:.2}" # IF less than :> Blue
            #color=(1,0,0,0.5)
            color=(1,0,0,0.5)

           # color=(0,0.5,0.9,0.5)
#
        d2d.SetColour(color)
        size = float(size) 
        d2d.DrawArc(Point2D(x,y), size, 0.0, 360.0)
        d2d.SetDrawOptions(d2d.drawOptions())
        d2d.SetFontSize(textsize)
        d2d.SetDrawOptions(d2d.drawOptions())

        d2d.SetColour((0,0,0))
        d2d.DrawString(text, Point2D(x,y))

    for step in range(len(test_path)):
        feature, threshold, direction = test_path[step]
        
        property_type, centroid_index = get_feature_property(int(feature))
        feature_value = X[feature]
        centroid_value = centroids[centroid_index]
        coords = get_coords(mol)
        distances = np.linalg.norm(centroid_value - coords, axis = 1)

        feature_class = property_type

        feature_name = get_feature_name(feature_class)
        if verbose:
            print(test_path[step])
            print(distances.min())
            print(feature_class)
        if feature_class == 0:
            draw_logP(centroid_value, d2d, threshold, feature_value >= threshold)
        elif feature_class == 1:
            draw_HA(centroid_value, d2d, feature_value >= threshold)
        elif feature_class == 2:
            draw_HD(centroid_value, d2d, feature_value >= threshold)
        elif feature_class == 3:
            if distances.min() < allow_dis:
                draw_checkmark(centroid_value, d2d)
            else:
                draw_X(centroid_value, d2d)
        else:
            print('err',feature_class)
        d2d.FinishDrawing()
        d2d.GetDrawingText()
        bio = BytesIO(d2d.GetDrawingText())
        img = Image.open(bio)
    return img

    

def custom_metric(decision_tree, centroids, X_test, X_mol, eps):
    from scipy.linalg import norm
    tree_ = decision_tree.tree_
    X_test = X_test.reshape(1,-1)
    test_path = decision_tree.decision_path(X_test).toarray()
    node_frequencies_screen = np.zeros(tree_.node_count, dtype=int)

    for sample in test_path:
        node_frequencies_screen += sample
    visited_nodes = {node_id for node_id in range(tree_.node_count) if node_frequencies_screen[node_id] > 0}
    positions = get_coords(X_mol)
    
    feature_index = [tree_.feature[node] for node in list(visited_nodes)]
    centroid_positions = []
    for feature in feature_index:
        if feature != _tree.TREE_UNDEFINED:
            centroid = get_feature_name(feature)
            centroid = int(centroid[-3:-1])
          #  print(centroid)
            centroid_positions.append(centroids[centroid])
    dis = []
    #for position in positions:
    #    for centroid in centroid_positions:
    #        dis.append(norm([position, centroid])/len(visited_nodes))
    dis = []
    for position in positions:
        for centroid in centroid_positions:
            dist = norm(np.array(position) - np.array(centroid))
            if dist < eps:
                dis.append(1)
            elif dist < 2:
                dis.append(1/(dist/eps)) # 1/(dist/eps) - when dist = eps, 1/1, else 
            else:
                dis.append(0)
    norm_dis = np.array(dis)

    adjusted_distance = np.sum(norm_dis)

    return adjusted_distance
  #  return sorted(dis)[:len(visited_nodes)]

def trace_path(final_tree, X_train, X_screen, indices, mols_train, hit_number, true_id = None, true_index = False, return_all = False):
    
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

def get_distances(final_tree, test_object, molecule, descriptors, centroids, numb_atom=1):

    path_info = get_test_path(final_tree, test_object)
    temp_dis = []
    distances_from_centroids = {}
    for feature, threshold, direction in path_info:
        property_type, centroid_index = get_feature_property(int(feature))

        if centroid_index >= len(centroids):
            continue  

        centroid = centroids[centroid_index]
        coords = get_coords(molecule)
        distances = np.linalg.norm(centroid - coords, axis=1)
        closest_atoms = np.argsort(distances)[:numb_atom]

        distances_from_centroids[feature] = distances[closest_atoms].tolist()
        temp_dis.append(distances[closest_atoms].tolist())
    me_dis = np.mean(temp_dis)
    #me_dis = np.mean(distances_from_centroids[feature] for feature, _,_ in path_info)
    return distances_from_centroids, me_dis

def plot_image_grid(images, cols=3, title=None, dpi = 500):
    rows = (len(images) + cols - 1) // cols  
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), dpi = dpi)

    axes = axes.flatten() if rows > 1 else [axes]
    #print([type(img) for img in images])

    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i])
            ax.axis("off") 
        else:
            ax.axis("off")  
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    #plt.savefig('/home/henry/Downloads/fig.png', dpi = 500)
    plt.show()
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

def get_data(proper, value, pos, choose_prop):
    prop_df = pd.DataFrame()
    prop_df['x'] = [pos[i][0] for i in range(len(pos))]
    prop_df['y'] = [pos[i][1] for i in range(len(pos))]
    prop_df['value'] = value
    prop_df['property'] = proper
   # print(len(cont_df))
    prop_df = prop_df[prop_df['property']==choose_prop]
    prop_df.sort_values(by='x', inplace=True)
    mean_rows = []
    rows = []
    current_x = None
    current_y = None

    for i, row in prop_df.iterrows():
        if row['x'] != current_x:
            if rows:  
                mean_value = np.mean(rows)
                mean_rows.append({'x': current_x, 'y': current_y, 'mean_value': mean_value, 'rows': rows})
            rows = [row['value']]
            current_x = row['x']
            current_y = row['y'] 
        else:
            rows.append(row['value'])
    if rows:
        mean_value = np.mean(rows)
        mean_rows.append({'x': current_x, 'y': current_y, 'mean_value': mean_value, 'rows': rows})
    rows = pd.DataFrame(mean_rows)
    return rows

def leave_one_out_cv(mols_train, IC50, X, centroids, eps):
    loo = LeaveOneOut()
    true_IC50 = np.zeros(len(IC50))
    pred_IC50 = np.zeros(len(IC50))
    mols_list = []
    trees_ = []
    metr = []
    #coun = np.zeros(len(IC50))
    #epsss = np.zeros(len(IC50))
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = IC50[train_index], IC50[test_index]
        mol_test = mols_train[test_index[0]]
        
        final_tree = DecisionTreeClassifier(max_depth=None, random_state=0)
        final_tree.fit(X_train, y_train)
        trees_.append(final_tree)
        y_pred = final_tree.predict(X_test)[0]
        true_IC50[test_index] = y_test[0]; pred_IC50[test_index] = y_pred; mols_list.append(mol_test)
        metric = custom_metric(final_tree, centroids, X_test, mol_test, eps)
        metr.append(metric)
    return true_IC50, pred_IC50,  mols_list, trees_, metr
import gc
###
### New function for leave one out cross validation - multiconformational
#def leave_one_out_cv(mols_train, IC50, identity, X_train, centroids, eps):
#    identity = np.array(identity)  
#    print('using multiconformational')
#    true_IC50 = IC50
#    pred_IC50 = np.zeros(len(IC50))
#    mols_list = []
#    trees_ = []
#    metr = []
#
#    unique_ids = np.unique(identity)
#    for i in unique_ids:
#        mask = identity != i
#        nonmask = identity == i
#
#        X_train_loo = X_train[mask]
#        y_train = IC50[mask]
#        X_test_loo = X_train[nonmask]
#        y_test = IC50[nonmask]
#
#        if X_test_loo.shape[0] == 0:
#            print(f"[ERROR] No test instances found for identity {i} — identity array might not match expectations.")
#            continue
#
#        X_test_loo = X_test_loo[0].reshape(1, -1)
#        y_test = y_test[0]
#        mol_test = mols_train[i]
#
#        final_tree = DecisionTreeClassifier(max_depth=None, random_state=0)
#        final_tree.fit(X_train_loo, y_train)
#        trees_.append(final_tree)
#
#        y_pred = final_tree.predict(X_test_loo)[0]
#        #true_IC50[i] = y_test
#        pred_IC50[i] = y_pred
#        mols_list.append(mol_test)
#
#        metric = custom_metric(final_tree, centroids, X_test_loo[0], mol_test, eps)
#        metr.append(metric)
#
#    return true_IC50, pred_IC50, mols_list, trees_, metr
def evaluate_combination(mols_train, mols_screen, IC50, params, idx, total):
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
    true_IC50, pred_IC50, mols_list, trees_, metr = leave_one_out_cv(mols_train, IC50, X_train, centroids, eps)
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

def retrieve_matching_mols_from_node(decision_tree, X_train, mols_train, mol_test, X_test, centroids, alt_true_IC50, eps, allow_dis, cols=1, size=(400,400), only_actives=False):
    tree_ = decision_tree.tree_
    X_test = X_test.reshape(1,-1)
    test_path = decision_tree.decision_path(X_test).toarray()
    train_path = decision_tree.decision_path(X_train).toarray()
    print(test_path, np.shape(test_path))
    node_frequencies_screen = np.zeros(tree_.node_count, dtype=int)

    for sample in test_path:
        node_frequencies_screen += sample
    visited_nodes = {node_id for node_id in range(tree_.node_count) if node_frequencies_screen[node_id] > 0}
    k=0
    Mol_matches = []; X_matches = []; IC50_matches = []
    #print(visited_nodes)
    #Mol_matches.append(mol_test); X_matches.append(X_test); IC50_matches.append('hit')
    for i, path in enumerate(train_path):
        node_frequencies_train = np.zeros(tree_.node_count, dtype=int)
        node_frequencies_train += path
        nodes = {node_id for node_id in range(tree_.node_count) if node_frequencies_train[node_id] > 0}
        if nodes == visited_nodes:
            if k == 0:
                Mol_matches.append(mol_test)
                X_matches.append(X_test[0])
                IC50_matches.append('hit')
            if only_actives:
                if k == 0:
                    Mol_matches.append(mol_test)
                    X_matches.append(X_test[0])
                    IC50_matches.append('hit')
                
                if alt_true_IC50[i] == '>100':
                    continue
                    
                if float(alt_true_IC50[i]) < 40:
                    IC50_matches.append(alt_true_IC50[i])
                    X_matches.append(X_train[i])
                    Mol_matches.append(mols_train[i])
                    
                else:
                    continue
                
            
            else:
                k+=1
                #print('match',k, list(nodes)[-1])
                Mol_matches.append(mols_train[i])
                X_matches.append(X_train[i])
                IC50_matches.append(alt_true_IC50[i])
    print(f"Found {len(X_matches)} matching molecules.")
    print(f"X_matches shape {np.shape(X_matches)}")
    #visualize(final_tree, X_test, mol_test, X_train, centroids, draw=True, textanot=True, tex = f"test molecule \n pred. active {What_mol}", size=size)
    images = [visualize(decision_tree, X_matches[i], Mol_matches[i], X_train, centroids, eps, allow_dis, draw=True, textanot=True, tex = f"ic50 {IC50_matches[i]:.4}", size=size) for i in range(len(X_matches))]
    
    return plot_image_grid(images, cols = cols)

def trace_path_pic(final_tree, X_train, X_screen, mols_train, hit_number, alt_true_IC50, true_id = None, true_index = False, return_all = False, verbose = True):
    if true_index == False:
        path_info = get_test_path(final_tree, X_screen[hit_number].reshape(-1,1))
    if true_index:
        path_info = get_test_path(final_tree, X_screen[true_id].reshape(-1,1))
    
    shared_path1 = []
    shared_path0 = []
    #print(path_info)
    decision_paths = final_tree.decision_path(X_train).toarray()

    feature_id_1 = path_info[-1][0]
    centroid_id_0 = path_info[-1][0]
    if verbose:
        print('feature', feature_id_1)
    splitting_nodes_1 = np.where(final_tree.tree_.feature == feature_id_1)[0]

    if len(splitting_nodes_1) == 0:
        
        print(f"Feature {feature_id_1} not found in decision tree splits.")
        return [], None 
    if verbose:
        print(f"Nodes splitting on Feature {feature_id_1}: {splitting_nodes_1}, centroid {centroid_id_0 // 4}")

    samples_splitting_at_feature_1 = np.where(decision_paths[:, splitting_nodes_1].sum(axis=1) > 0)[0]
    if verbose: 
        print(f"Samples splitting at Feature {feature_id_1}: {samples_splitting_at_feature_1[:20]}")

    shared_path1.extend(samples_splitting_at_feature_1)
    
    feature_id_0 = path_info[-2][0]
    
    splitting_nodes_0 = np.where(final_tree.tree_.feature == feature_id_0)[0]
    if verbose:
        print(f"Nodes splitting on Feature {feature_id_0}: {splitting_nodes_0}")

    
    samples_splitting_at_feature_0 = np.where(decision_paths[:, splitting_nodes_0].sum(axis=1) > 0)[0]
    if verbose:
        print(f"Samples splitting at Feature {feature_id_0}: {samples_splitting_at_feature_0[:20]}")
    shared_path0.extend(samples_splitting_at_feature_0)

    common_samples = list(set(shared_path1) & set(shared_path0))
    if verbose:
        print(f"Common samples (indices in X_train): {common_samples}")
    
    molecules_splitting_at_feature = [mols_train[idx] for idx in common_samples]
    x_molecules_spl = [X_train[idx] for idx in common_samples]
    ic50_mols_splitting_at_feature = [alt_true_IC50[idx] for idx in common_samples]
    if verbose:
        print(f"Found {len(molecules_splitting_at_feature)} molecules following the common prior path.")
    selected_mols = [molecules_splitting_at_feature[idx] for idx in range(len(molecules_splitting_at_feature))]
    legends = [f'IC50: {ic50_mols_splitting_at_feature[idx]}' for idx in range(len(selected_mols))]
    img = Draw.MolsToGridImage(selected_mols, molsPerRow=4, subImgSize=(600, 600), maxMols=25, legends = legends) 
    if return_all:
        return molecules_splitting_at_feature, img, common_samples, X_screen[hit_number].reshape(-1,1), hit_number
    return common_samples, img

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
    acc = [alt_true_IC50[i] for i in train_indices]

    mols_screen_node = [mols_screen[i] for i in screen_indices]
    X_train_node = [X_train[i] for i in train_indices]
    X_screen_node = [X_screen[i] for i in screen_indices]

    return mols_train_node, mols_screen_node, X_train_node, X_screen_node, class_train_node, acc, screen_indices, train_indices
def draw_trainmolecules_from_node(decision_tree, X_train, mols_train, node_id, IC50, 
                                  alt_true_IC50, centroids, eps, allow_dis, override = None
                                ):
    """
    Retrieves molecules from a specific node of the tree.
    """
    decision_paths_train = decision_tree.decision_path(X_train).toarray()
   # print(decision_paths_train)
    # decision_paths_screen = decision_tree.decision_path(X_screen).toarray()
    #print(np.shape(decision_paths_train))
    #print(decision_paths_train[:,   node_id] > 0)
    train_indices = np.where(decision_paths_train[:,   node_id] > 0)[0]
    if len(train_indices) == 0:
        print(f"No molecules found in node {node_id}.")
        return None
    #print('train:',train_indices)
    #print('test', screen_indices)
   #mols_train_node = [mols_train[i] for i in train_indices]
   #class_train_node = [IC50[i] for i in train_indices]
    acc = [alt_true_IC50[i] for i in train_indices]
    
    #  mols_screen_node = [mols_screen[i] for i in screen_indices]
    X_train_node = [X_train[i] for i in train_indices]
    # X_screen_node = [X_screen[i] for i in screen_indices]
    #images = [visualize(decision_tree, X_train[i], mols_train[i], X_train, centroids, eps, allow_dis, draw=True, textanot=True, tex = f"index {i}, ic50: {alt_true_IC50[i]:.4}", font=20, size = (900,900), textsize = 20) for i in train_indices]
    col_leng = len(train_indices) if len(train_indices) < 4 else 3
    imagelen = len(train_indices)
    if len(train_indices) >= 4:
        imagelen = col_leng+1
    if len(train_indices) < 3:
        col_leng = 1
    if override != None:
        imagelen = override
        col_leng = 5
        #imagelen = len(train_indices)
    images = [visualize(decision_tree, X_train[i], mols_train[i], X_train, centroids, eps, allow_dis, textanot=True, tex = f"index {i}, ic50: {alt_true_IC50[i]:.4}", font=20, size = (900,900), textsize = 20) for i in train_indices[:imagelen]]

    plot_image_grid(images, cols = col_leng, dpi = 500)

def build_tree_graph_labeled(decision_tree, mols_train, X_train, mols_screen, X_screen, IC50, alt_true_IC50,
                             density_scaling_factor=2000, color_intensity_factor=1.0):
    tree_ = decision_tree.tree_
    
    decision_paths_train = decision_tree.decision_path(X_train).toarray()
    decision_paths_screen = decision_tree.decision_path(X_screen).toarray()
    print(density_scaling_factor)
    if not isinstance(density_scaling_factor, (int, float)):
        raise TypeError(f"density_scaling_factor must be a number, but got {type(density_scaling_factor)}")

    node_frequencies_train = np.zeros(tree_.node_count, dtype=int)
    node_frequencies_screen = np.zeros(tree_.node_count, dtype=int)

    for sample in decision_paths_train:
        node_frequencies_train += sample
    for sample in decision_paths_screen:
        node_frequencies_screen += sample

    max_freq = max(max(node_frequencies_train), max(node_frequencies_screen))
    if max_freq == 0:
        max_freq = 1  # Avoid division by zero

    visited_nodes = {node_id for node_id in range(tree_.node_count) if node_frequencies_screen[node_id] > 0}

    if not visited_nodes:
        raise ValueError("No nodes were visited by X_screen molecules.")

    G = nx.DiGraph()
    node_sizes = {}
    node_colors = {}
    node_labels = {}
    node_class = {}

    for node_id in visited_nodes:
        left_child = tree_.children_left[node_id]
        right_child = tree_.children_right[node_id]
        if left_child and right_child == -1:
            mols_train_node, _, _, _, temp_IC50,_,_,_ = retrieve_molecules_from_node(decision_tree, X_train, X_screen, mols_train, mols_screen, node_id, IC50, alt_true_IC50)
            node_class[node_id] = temp_IC50[0]
            #print(node_id, temp_IC50)
        else:
            node_class[node_id] = -1
        if left_child in visited_nodes:
            G.add_edge(node_id, left_child)
        if right_child in visited_nodes:
            G.add_edge(node_id, right_child)
    density_scaling_factor = 3000
    for node_id in visited_nodes:
        G.add_node(node_id)

        visit_count_train = node_frequencies_train[node_id]
        visit_count_screen = node_frequencies_screen[node_id]

        node_sizes[node_id] = 500 + (density_scaling_factor * ((visit_count_train + visit_count_screen) / max_freq))

        node_colors[node_id] = ((visit_count_train + visit_count_screen) / max_freq) ** color_intensity_factor


        def get_feature_name(feature_index):
            feature_map = {0: "logP", 1: "HBA", 2: "HBD", 3: "Atom"}  
            property_type, centroid_index = get_feature_property(feature_index)

            feature_name = feature_map.get(property_type, "Unknown")  
            return feature_name, centroid_index
        feature_index = tree_.feature[node_id]
        #print(node_labels)
        #print(G.nodes)

        if feature_index != _tree.TREE_UNDEFINED:
 
            feature_name,  centroid= get_feature_name(feature_index)
            node_labels[node_id] = (f"Feature {feature_index}\n Centroid {centroid} \n{feature_name}\n" +
                                    f"Train Visits: {visit_count_train} \n | Screen Visits: {visit_count_screen}\n" 
                                    )
        else:
            node_labels[node_id] = (f"Leaf Node\nTrain Visits: {visit_count_train} \n | Screen Visits: {visit_count_screen}\nNode: {node_id}\n"+
                                    f"Class: {'Active' if node_class[node_id] == 1 else 'Inactive'}"
                              )


    return G, node_sizes, node_colors, node_labels
def plot_topological_decision_tree_labeled(decision_tree, mols_train, X_train, mols_screen, X_screen, feature_names, alt_true_IC50,
                                           min_samples, eps,
                                           density_scaling_factor=3000, color_intensity_factor=2.0, 
                                           text_color="magenta", label_offset=(0.5, 0.5),
                                           text_bg_color="white", text_bg_alpha=0.7):
     
    G, node_sizes, node_colors, node_labels = build_tree_graph_labeled(decision_tree, mols_train, X_train, mols_screen, X_screen, feature_names, alt_true_IC50, 
                                                                       density_scaling_factor, color_intensity_factor)
    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot")

    cmap = plt.cm.Blues
    color_values = [cmap(node_colors[node]) for node in G.nodes]
    
    plt.figure(figsize=(18, 18), dpi=100)
    nx.draw(G, pos, with_labels=False, node_size=[node_sizes[n] for n in G.nodes], 
            node_color=color_values, edge_color="gray", linewidths=1, cmap=cmap, alpha=0.7)

    label_pos = {node: (pos[node][0] + label_offset[0], pos[node][1] + label_offset[1]) for node in G.nodes}

    for node, (x, y) in label_pos.items():

        plt.text(x, y, node_labels[node], fontsize=12, fontweight="bold", color=text_color,
                 bbox=dict(facecolor=text_bg_color, alpha=text_bg_alpha, edgecolor='none', boxstyle="round,pad=0.3"))

    plt.title(f"Decision Tree Traversal by X_screen Molecules Using DBscan, min_samples = {min_samples}, eps = {eps}", fontsize=25)
    plt.savefig("figure.pdf", format='pdf', bbox_inches='tight')
    plt.show()

def draw_node(index, number_of_train_mols, final_tree, X_train, 
              X_screen, mols_train, mols_screen,centroids, IC50, 
              alt_true_IC50, eps, allow_dis, return_hits = False, pos = [], custom_ids = False, order = [],
                max_plots = 15, size = (800,800), tex_pos=Point2D(0,4), textsize=15, baseFont=2, n_cols = 2, save = False):
    '''
    index: Leaf node number from decision tree (final_tree)
    number_of_train_mols: Number of training molecules to display
    final_tree: Trained decision tree
    X_train: training molecules descriptors
    X_screen: screening molecules descriptors
    mols_train: training rdkit molobjects
    mols_screen: screening rdkit molobjects
    centroids: centroids used in forming the train and screening descriptors
    IC50: class labels for the training molecules
    alt_true_IC50: the measured ic50 label
    eps: epsilon; maximum distance between two points for them to be considered neighbours. Parameter used producing centroids and thereby descriptors
    allow_dis: maximum distance of scaled-down contribution to descriptor
    return_hits: Whether to return additional details such as the descriptors and their metrics
    pos: number of positive counts during repeated rand_state screening
    max_plots: maximum number of plots to display
    size: size of plots
    tex_pos: Point2D value for the placement of the labels
    textsize: Size of the text
    baseFont: scale of the text
    n_cols: Number of columns to show the molecules in
    save: Whether to save the plot or not
    '''
    
    mols_train_node, mols_screen_node, X_train_node, X_screen_node, _, ic, screen_id, train_id = retrieve_molecules_from_node(
        final_tree, X_train, X_screen, mols_train, mols_screen, index, IC50, alt_true_IC50
    )

    print(f"Number of train mols: {len(mols_train_node)}, Number of screen mols: {len(mols_screen_node)}")

    train_sample, train_molecule = X_train_node, mols_train_node
    #lim = 15
    #num_plots = len(X_screen_node) + 1 if len(X_screen_node) < lim else lim - 1

    met = [
        custom_metric(final_tree, centroids, X_screen_node[i], mols_screen_node[i], eps)
        for i in range(len(X_screen_node))
    ]
    def tanimoto_similarity(a: np.ndarray, b: np.ndarray) -> float:
        dot_prod = np.dot(a, b)
        norm_a = np.dot(a, a)
        norm_b = np.dot(b, b)

        denom = norm_a + norm_b - dot_prod
        if denom == 0:
            return 0.0 
        return dot_prod / denom
    similarity_score = []
    for X_s in X_screen_node:
        temp_sim = []
        for X_t in X_train_node:
            tan_sim = tanimoto_similarity(X_t, X_s)
            temp_sim.append(tan_sim)
        similarity_score.append(np.average(temp_sim))

    temp_df = pd.DataFrame()
    temp_df['mols_sc'] = mols_screen_node
    temp_df['X_mols'] = X_screen_node
    temp_df['met'] = met
    temp_df['similarity'] = similarity_score
    temp_df['id'] = screen_id

    if custom_ids:
        temp_df = temp_df[temp_df['id'].isin(order)]

    temp_df = temp_df.sort_values(by='similarity', ascending=False).reset_index(drop=True)

    mols_screen_node = temp_df['mols_sc']; X_screen_node = temp_df['X_mols']
    screen_id = temp_df['id']

    k=0
    num_train = [number_of_train_mols if len(mols_train_node) > number_of_train_mols else len(mols_train_node)][0]
    
    perc_hit = []
    if len(pos) > 0:
        for i in range(len(mols_screen_node)):
            perc_hit.append(pos[screen_id[i]])
    upper_bound = len(mols_screen_node) if custom_ids == False else len(order)
    
    #NEED TO RESTRUCTURE BELOW...
    corr_max_plots = min(num_train + upper_bound, max_plots)
    n_rows = (corr_max_plots // n_cols) + (1 if corr_max_plots % n_cols > 0 else 0)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    axes = axes.flatten()  
    for i in range(corr_max_plots):
                if i < corr_max_plots - num_train and i < upper_bound:
                    if len(pos) != 0:
                        img = visualize(final_tree, X_screen_node[i], mols_screen_node[i],
                                        centroids, eps, allow_dis, textanot=True,
                                        tex=f"id: {screen_id[i]}",#Screen hit metric: {temp_df['similarity'].iloc[i]:.2f} pos: {pos[screen_id[i]]} 
                                        size=size, baseFont=baseFont, tex_pos=tex_pos, textsize=textsize)#, #pos: {positive_counts[screen_id[i]]}
                    else:
                        img = visualize(final_tree, X_screen_node[i], mols_screen_node[i],
                                        centroids, eps, allow_dis, textanot=True,
                                        tex=f"{temp_df['id'].iloc[i]}", #Screen hit metric: {temp_df['similarity'].iloc[i]:.2f}",
                                        size=size, baseFont=baseFont, tex_pos=tex_pos, textsize=textsize)#, #pos: {positive_counts[screen_id[i]]}
    
                else:
                    try:
                        img = visualize(final_tree, train_sample[k], train_molecule[k],
                                    centroids, eps, allow_dis, textanot=True, 
                                    tex=f"{train_id[k]} train molecule ic50 {float(ic[k]):.4}",# metric: {float(custom_metric(final_tree, centroids, train_sample[k], train_molecule[k], eps)):.2}",
    
                                    size=size, baseFont=baseFont, tex_pos=tex_pos, textsize=textsize)
                        k+=1
                    except:
                        img = visualize(final_tree, train_sample[k], train_molecule[k],
                                    centroids, eps, allow_dis, textanot=True, 
                                    tex=f"{train_id[k]} train molecule ic50 {ic[k]}",#metric: {float(custom_metric(final_tree, centroids, train_sample[k], train_molecule[k], eps)):.2}",
    
                                    size=size, baseFont=baseFont, tex_pos=tex_pos, textsize=textsize)
                        k+=1
                axes[i].imshow(img)
                axes[i].axis("off")


    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(os.getcwd(), 'saved_mols'), dpi = 1000)
    plt.show()

    if return_hits:
        try:
            if isinstance(perc_hit, list):
                return mols_screen_node, mols_train_node, X_screen_node, np.sort(met)[::-1], perc_hit
        except:
            return mols_screen_node, mols_train_node, X_screen_node, np.sort(met)[::-1], []
    #return temp_df
def draw_node(index, number_of_train_mols, final_tree, X_train, 
              X_screen, mols_train, mols_screen,centroids, IC50, 
              alt_true_IC50, eps, allow_dis, return_hits = False, pos = [], custom_ids = False, order = [],
                max_plots = 15, size = (800,800), tex_pos=Point2D(0,4), textsize=15, baseFont=2, n_cols = 2, save = False):
    '''
    index: Leaf node number from decision tree (final_tree)
    number_of_train_mols: Number of training molecules to display
    final_tree: Trained decision tree
    X_train: training molecules descriptors
    X_screen: screening molecules descriptors
    mols_train: training rdkit molobjects
    mols_screen: screening rdkit molobjects
    centroids: centroids used in forming the train and screening descriptors
    IC50: class labels for the training molecules
    alt_true_IC50: the measured ic50 label
    eps: epsilon; maximum distance between two points for them to be considered neighbours. Parameter used producing centroids and thereby descriptors
    allow_dis: maximum distance of scaled-down contribution to descriptor
    return_hits: Whether to return additional details such as the descriptors and their metrics
    pos: number of positive counts during repeated rand_state screening
    max_plots: maximum number of plots to display
    size: size of plots
    tex_pos: Point2D value for the placement of the labels
    textsize: Size of the text
    baseFont: scale of the text
    n_cols: Number of columns to show the molecules in
    save: Whether to save the plot or not
    '''
    
    mols_train_node, mols_screen_node, X_train_node, X_screen_node, _, ic, screen_id, train_id = retrieve_molecules_from_node(
        final_tree, X_train, X_screen, mols_train, mols_screen, index, IC50, alt_true_IC50
    )

    print(f"Number of train mols: {len(mols_train_node)}, Number of screen mols: {len(mols_screen_node)}")

    train_sample, train_molecule = X_train_node, mols_train_node
    #lim = 15
    #num_plots = len(X_screen_node) + 1 if len(X_screen_node) < lim else lim - 1

    met = [
        custom_metric(final_tree, centroids, X_screen_node[i], mols_screen_node[i], eps)
        for i in range(len(X_screen_node))
    ]
    def tanimoto_similarity(a: np.ndarray, b: np.ndarray) -> float:
        dot_prod = np.dot(a, b)
        norm_a = np.dot(a, a)
        norm_b = np.dot(b, b)

        denom = norm_a + norm_b - dot_prod
        if denom == 0:
            return 0.0 
        return dot_prod / denom
    similarity_score = []
    for X_s in X_screen_node:
        temp_sim = []
        for X_t in X_train_node:
            tan_sim = tanimoto_similarity(X_t, X_s)
            temp_sim.append(tan_sim)
        similarity_score.append(np.average(temp_sim))

    temp_df = pd.DataFrame()
    temp_df['mols_sc'] = mols_screen_node
    temp_df['X_mols'] = X_screen_node
    temp_df['met'] = met
    temp_df['similarity'] = similarity_score
    temp_df['id'] = screen_id

    if custom_ids:
        temp_df = temp_df[temp_df['id'].isin(order)]

    temp_df = temp_df.sort_values(by='similarity', ascending=False).reset_index(drop=True)

    mols_screen_node = temp_df['mols_sc']; X_screen_node = temp_df['X_mols']
    screen_id = temp_df['id']

    k=0
    num_train = [number_of_train_mols if len(mols_train_node) > number_of_train_mols else len(mols_train_node)][0]
    
    perc_hit = []
    if len(pos) > 0:
        for i in range(len(mols_screen_node)):
            perc_hit.append(pos[screen_id[i]])
    upper_bound = len(mols_screen_node) if custom_ids == False else len(order)
    
    #NEED TO RESTRUCTURE BELOW...
    corr_max_plots = min(num_train + upper_bound, max_plots)
    n_rows = (corr_max_plots // n_cols) + (1 if corr_max_plots % n_cols > 0 else 0)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    axes = axes.flatten()  
    for i in range(corr_max_plots):
                if i < corr_max_plots - num_train and i < upper_bound:
                    if len(pos) != 0:
                        img = visualize(final_tree, X_screen_node[i], mols_screen_node[i],
                                        centroids, eps, allow_dis, textanot=True,
                                        tex=f"id: {screen_id[i]}", #Screen hit metric: {temp_df['similarity'].iloc[i]:.2f} pos: {pos[screen_id[i]]} 
                                        size=size, baseFont=baseFont, tex_pos=tex_pos, textsize=textsize)#, #pos: {positive_counts[screen_id[i]]}
                    else:
                        img = visualize(final_tree, X_screen_node[i], mols_screen_node[i],
                                        centroids, eps, allow_dis, textanot=True,
                                        tex=f"{temp_df['id'].iloc[i]}",# Screen hit metric: {temp_df['similarity'].iloc[i]:.2f}",
                                        size=size, baseFont=baseFont, tex_pos=tex_pos, textsize=textsize)#, #pos: {positive_counts[screen_id[i]]}
    
                else:
                    try:
                        img = visualize(final_tree, train_sample[k], train_molecule[k],
                                    centroids, eps, allow_dis, textanot=True, 
                                    tex=f"{train_id[k]} ic50 {float(ic[k]):.4}",
    
                                    size=size, baseFont=baseFont, tex_pos=tex_pos, textsize=textsize)
                        k+=1
                    except:
                        img = visualize(final_tree, train_sample[k], train_molecule[k],
                                    centroids, eps, allow_dis, textanot=True, 
                                    tex=f"{train_id[k]} ic50 {ic[k]}",
    
                                    size=size, baseFont=baseFont, tex_pos=tex_pos, textsize=textsize)
                        k+=1
                axes[i].imshow(img)
                axes[i].axis("off")


    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(os.getcwd(), 'saved_mols'), dpi = 2000)
    plt.show()

    if return_hits:
        try:
            if isinstance(perc_hit, list):
                return mols_screen_node, mols_train_node, X_screen_node, np.sort(met)[::-1], perc_hit
        except:
            return mols_screen_node, mols_train_node, X_screen_node, np.sort(met)[::-1], []
def train_and_predict_100(i, IC50, mols_train, X_train, mols_screen, X_screen, centroids, eps):

    final_tree_100 = DecisionTreeClassifier(max_depth=None, random_state=i)
    final_tree_100.fit(X_train, IC50)
    preds = final_tree_100.predict(X_screen)
    metrics = []
    for i in range(len(mols_screen)):
        m = custom_metric(final_tree_100, centroids, np.array(X_screen[i]), mols_screen[i], eps)
        metrics.append(m)

    return preds, final_tree_100, metrics
import time

# Make descriptors fast:
def make_descriptors_fast(mols_train, mols_screen, n_clusters, eps, allow_dis, DBScan_mode = False, min_samples=None, distance_based=False, return_centroids = False): 
    start_time = time.time()
    def extract_2D_coordinates(mols_train):
        total_atoms = sum(m.GetNumAtoms() for m in mols_train)
        coords = np.empty((total_atoms, 2), dtype=np.float32)

        idx = 0
        for m in mols_train:
            conf = m.GetConformer(0)
            natoms = m.GetNumAtoms()
            for i in range(natoms):
                pos = conf.GetAtomPosition(i)
                coords[idx] = [pos.x, pos.y]
                idx += 1
        return coords
    
    X = extract_2D_coordinates(mols_train)

    def DBScan(X, n_clusters, eps=eps):
        if DBScan_mode == False:
            agglom = AgglomerativeClustering(n_clusters=n_clusters)
            labels = agglom.fit_predict(X)
        if DBScan_mode == True:
            print('DBscan clustering')
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
        unique_labels = set(labels)
        clusters = {label: X[labels == label] for label in unique_labels if label != -1}
        centroids = {label: cluster.mean(axis=0) for label, cluster in clusters.items()}
        centroids = np.stack(list(centroids.values()))
        return centroids

    centroids = DBScan(X, n_clusters)
    #print('centroids calculated')
    def preprocess_mols(mols):
        coords_list = []
        logP_list = []
        HBA_list = []
        HBD_list = []
        for m in mols:
            m = Chem.RemoveAllHs(m)
            coords = get_coords(m)  # (n_atoms, 2)
            logP = get_logP(m)      # (n_atoms,)
            HBDs, HBAs = get_HBs(m)  # (list of HBDs, list of HBAs)

            HBA_mask = np.zeros(len(logP), dtype=np.int32)
            HBD_mask = np.zeros(len(logP), dtype=np.int32)

            # Fill masks
            for i in range(len(logP)):
                if i in HBAs:
                    HBA_mask[i] = 1
                if i in HBDs:
                    HBD_mask[i] = 1

            coords_list.append(coords)
            logP_list.append(logP)
            HBA_list.append(HBA_mask)
            HBD_list.append(HBD_mask)
        max_atoms = max([len(coords) for coords in coords_list])

        for i in range(len(mols)):
            coords_list[i] = np.pad(coords_list[i], ((0, max_atoms - len(coords_list[i])), (0, 0)), 'constant')
            logP_list[i] = np.pad(logP_list[i], (0, max_atoms - len(logP_list[i])), 'constant')
            HBA_list[i] = np.pad(HBA_list[i], (0, max_atoms - len(HBA_list[i])), 'constant')
            HBD_list[i] = np.pad(HBD_list[i], (0, max_atoms - len(HBD_list[i])), 'constant')
        coords_array = np.array(coords_list, dtype=np.float32)
        logP_array = np.array(logP_list, dtype=np.float32)
        HBA_array = np.array(HBA_list, dtype=np.int32)
        HBD_array = np.array(HBD_list, dtype=np.int32)
   
        return coords_array, logP_array, HBA_array, HBD_array
    from numba import jit, prange
    @jit(nopython=True, parallel=True)
    def compute_descriptors_numba(coords_array, logP_array, HBA_array, HBD_array, centroids, eps, dis_allow):
        n_mols = coords_array.shape[0]
        n_centroids = centroids.shape[0]

        descriptors = np.zeros((n_mols, n_centroids * 4), dtype=np.float64)

        for mm in prange(n_mols):
            coords = coords_array[mm]
            logP = logP_array[mm]
            HBA = HBA_array[mm]
            HBD = HBD_array[mm]
            n_atoms = coords.shape[0]

            desc = np.zeros((n_centroids, 4), dtype=np.float64)

            for i in range(n_atoms):
                coord = coords[i]
                lp = logP[i]

                if np.isnan(lp) or np.isinf(lp):
                    continue

                dists = np.sqrt(np.sum((centroids - coord) ** 2, axis=1))

                for cluster_idx in range(n_centroids):
                    d = dists[cluster_idx]

                    if d > dis_allow: 
                        continue

                    if d > eps:
                        w = lp / (d / eps) 
                    else:
                        w = lp  
                    desc[cluster_idx, 0] += w
                    
                    w_hbact = 1.0 if d < eps else eps / d 
                    desc[cluster_idx, 1] += HBA[i] * (w_hbact) 
                    desc[cluster_idx, 2] += HBD[i] * (w_hbact) 
                    desc[cluster_idx, 3] = 1.0  

            for c in range(n_centroids):
                for f in range(4):
                    descriptors[mm, c * 4 + f] = desc[c, f]

        return np.round(descriptors, decimals=3)

    coords_array, logP_array, HBA_array, HBD_array = preprocess_mols(mols_train)

    X_train = compute_descriptors_numba(coords_array, logP_array, HBA_array, HBD_array, centroids, eps, allow_dis)
    coords_array, logP_array, HBA_array, HBD_array = preprocess_mols(mols_screen)
    X_screen = compute_descriptors_numba(coords_array, logP_array, HBA_array, HBD_array, centroids, eps, allow_dis)

    #X_train = compute_descriptors(mols_train, centroids, eps, allow_dis)
    #X_screen = compute_descriptors(mols_screen, centroids, eps, allow_dis)
    end_time = time.time()


    print(f"Made {len(mols_screen) + len(mols_train)} descriptors in {end_time - start_time} seconds {(len(mols_screen) + len(mols_train))*len(X_train[0])*64/(end_time - start_time):.2f} bits / s")    
    if return_centroids == True:
        return X_train, X_screen, centroids
    return X_train, X_screen