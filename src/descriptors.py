import numpy as np
from numba import jit, prange
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import time
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import itertools
from rdkit.Chem import AllChem

def get_coords(m: Chem.rdchem.Mol) -> np.ndarray:
    """
    Extract 2D atomic coordinates (x, y) from the first conformer of a molecule.    
    Parameters
    ----------
    m : rdkit.Chem.rdchem.Mol
        RDKit molecule object. Must contain at least one conformer. 
    Returns
    -------
    numpy.ndarray of shape (n_atoms, 2)
        Array of x, y coordinates for each atom in the molecule.
    """
    X = []
    for i, atom in enumerate(m.GetAtoms()):
        positions = m.GetConformer(0).GetAtomPosition(i)
        X.append([positions.x, positions.y])

    return np.array(X)

def get_HBs(m):
    NumHBD = Chem.MolFromSmarts("[N&!H0&v3,N&!H0&+1&v4,O&H1&+0,S&H1&+0,n&H1&+0]") 
    NumHBA = Chem.MolFromSmarts(
        "[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),"
        "$([O,S;H0;v2]),"
        "$([O,S;-]),"
        "$([N;v3;!$(N-*=!@[O,N,P,S])]),"
        "$([nH0,o,s;+0])]"
    )  
    # HBA and HBD from https://doi.org/10.1002/(SICI)1097-0290(199824)61:1<47::AID-BIT9>3.0.CO;2-Z
    #NumHBA = Chem.MolFromSmarts("[$([O,S;H1;v2]-[!$(*4[O,N,P,S])]),[O,S;H0;v2], [O,S;-],$([N&v3;H1,H2]-[!$(*4[O,N,P,S])]),[N;v3;H0],[n,o,s;+0],F]")
    #NumHBD = Chem.MolFromSmarts("[[N;!H0;v3],[N;!H0;+1;v4],[O,S;H1;+0],[n;H1;+0]]")
    HBDs = np.array(m.GetSubstructMatches(NumHBD)).flatten()
    HBAs = np.array(m.GetSubstructMatches(NumHBA)).flatten()

    return HBDs, HBAs

def get_logP(m):
    mol = Chem.Mol(m)

    mol = Chem.AddHs(mol)
    
    contribs = rdMolDescriptors._CalcCrippenContribs(mol)
    
    logP = [x for x, y in contribs] 

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            heavy_idx = atom.GetNeighbors()[0].GetIdx()
            logP[heavy_idx] += logP[atom.GetIdx()] 
    
    return logP[:mol.GetNumHeavyAtoms()]  

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


def draw_centroids(mols_train, min_samples,eps, draw = False, return_dat = False):
    # Making centroids
    X = []
    for m in mols_train:
        for i, atom in enumerate(m.GetAtoms()):
            positions = m.GetConformer(0).GetAtomPosition(i)
            X.append([positions.x, positions.y])
    X = np.array(X)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    
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

    if draw:
        fig = plt.figure(figsize=(10, 7))
        for label, cluster in clusters.items():
            plt.scatter(cluster[:, 0], cluster[:, 1], color=colors[label], marker=markers[label])#, label=f"Cluster {label}")
        for label, centroid in centroids.items():
            plt.scatter(centroid[0], centroid[1], color="black", marker="X", s=150)#, label=f"Centroid {label}")
        plt.title(f'DBSCAN Clustering (min_samples = {min_samples}, eps = {eps})')
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        return fig

    centroids = np.stack(list(centroids.values()))

    return centroids
def DBScan(X, min_samples, eps):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    unique_labels = set(labels)
    clusters = {label: X[labels == label] for label in unique_labels if label != -1}
    centroids = {label: cluster.mean(axis=0) for label, cluster in clusters.items()}
    centroids = np.stack(list(centroids.values()))
    return centroids


def make_descriptors_fast(mols_train, min_samples, eps, allow_dis, mols_screen = None, 
                          return_centroids = False, incl_screen_mols = False, verbose = True,
                          pre_calc_centroid = False, centroids = None):
     
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
    #Extracting 2D information from the training molecules.
    X = extract_2D_coordinates(mols_train)
    #Making centroids from the training molecules
    if pre_calc_centroid:
        print("Using precomputed centroids!")
    else:   
        centroids = DBScan(X, min_samples, eps)

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
    
    @jit(nopython=True, parallel=True)
    def compute_descriptors_numba(coords_array, logP_array, HBA_array, HBD_array, centroids, eps, dis_allow):
        n_mols = coords_array.shape[0]
        n_centroids = centroids.shape[0]

        descriptors = np.zeros((n_mols, n_centroids * 4), dtype=np.float64)

        for mm in prange(n_mols):   
            # 
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
    #print('debug', np.sum(coords_array), np.sum(logP_array), np.sum(HBA_array), np.sum(HBD_array))

    ### Preprocess is safe!

    X_train = compute_descriptors_numba(coords_array, logP_array, HBA_array, HBD_array, centroids, eps, allow_dis)


    if incl_screen_mols:
        coords_array, logP_array, HBA_array, HBD_array = preprocess_mols(mols_screen)
        X_screen = compute_descriptors_numba(coords_array, logP_array, HBA_array, HBD_array, centroids, eps, allow_dis)

    end_time = time.time()

    if incl_screen_mols:
        if verbose:
            print(f"Made {len(mols_screen) + len(mols_train)} descriptors in {end_time - start_time} seconds {(len(mols_screen) + len(mols_train))*len(X_train[0])*64/(end_time - start_time):.2f} bits / s")    
        if return_centroids == True:
            return X_train, X_screen, centroids
        else:
            return X_train, X_screen
    else:
        if verbose:
            print(f"Made {len(mols_train)} descriptors in {end_time - start_time} seconds {(len(mols_train))*len(X_train[0])*64/(end_time - start_time):.2f} bits / s")    
        if return_centroids == True:
            return X_train, centroids
        else:
            return X_train

def reorient(mol, p, ref_indol, conf=None):
    """
    Reorient mol by aligning its indole substructure to ref_indol.

    - mol: RDKit Mol object
    - p: Indole scaffold
    - ref_indol: List of RDKit Point3D reference coordinates
    - conf: RDKit Conformer object (must match mol)
    """
    if conf is None:
        conf = mol.GetConformer()

    if conf.GetNumAtoms() != mol.GetNumAtoms():
        raise ValueError(f"Confomer atom count mismatch: conf={conf.GetNumAtoms()}, mol={mol.GetNumAtoms()}")

    if not mol.HasSubstructMatch(p):
        raise ValueError('No substructure match found for indole scaffold.')

    match = mol.GetSubstructMatch(p)
    indol_pos = [conf.GetAtomPosition(i) for i in match]

    if len(indol_pos) != len(ref_indol):
        raise ValueError(f"Mismatch in number of points: indole={len(indol_pos)}, ref_indol={len(ref_indol)}")

    P = np.array([[pt.x, pt.y, pt.z] for pt in indol_pos])
    Q = np.array([[pt.x, pt.y, pt.z] for pt in ref_indol])

    def kabsch(P, Q):
        centroid_P = np.mean(P, axis=0)
        centroid_Q = np.mean(Q, axis=0) #Center of respective set of points
        P_centered = P - centroid_P #realigned to the centers
        Q_centered = Q - centroid_Q
        H = P_centered.T @ Q_centered
        U, S, Vt = np.linalg.svd(H) #singular value decomposition
        R = Vt.T @ U.T #optimal rotation matrix
        if np.linalg.det(R) < 0: #non singualr matrix
            Vt[-1, :] *= -1 
            R = Vt.T @ U.T
        t = centroid_Q - R @ centroid_P #apply optimal rotation_matrix to align sets
        return R, t
    
    # Realigns
    R, t = kabsch(P, Q)

    for i in range(conf.GetNumAtoms()):
        pt = conf.GetAtomPosition(i)
        xyz = np.array([pt.x, pt.y, pt.z])
        new_xyz = R @ xyz + t
        conf.SetAtomPosition(i, new_xyz)
