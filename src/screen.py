from descriptors import make_descriptors_fast
import os
import yaml
import argparse
from rdkit import Chem
import numpy as np
#from descriptors import make_the_alignments
from rdkit.Chem import AllChem
import pickle

def main():
    script_dir = os.getcwd()
    par_dir = os.path.dirname(script_dir)
    sdf_path = os.path.join(par_dir, "data/processed/toy_indoles_aligned.sdf")
    sdf_screen_path = os.path.join(par_dir, "data/screening/HIT_locator.sdf")
    
    parser = argparse.ArgumentParser(description="Please input your desired parameters for the descriptor generation!")
    #parser.add_argument("--eps", type=float, default=0.5, help="DBSCAN epsilon value")
    #parser.add_argument("--min_samples", type=int, default=2, help="DBSCAN min_samples")
    #parser.add_argument("--allow_dis", type=float, default=1.8, help="Allowed distance threshold")
    parser.add_argument("--sdf_t", type=str, default=sdf_path,
                        help="Path to input SDF file")
    parser.add_argument("--sdf_s", type=str, default=sdf_screen_path,
                        help="Path to screening SDF file")
    #parser.add_argument("--load_opt_params", action="store_true", default = False, help = "Load parameters from optimization")
    args = parser.parse_args()


    param_path = os.path.join(par_dir, "results/models/params.yaml")
    with open(param_path, 'r') as file:
        params = yaml.safe_load(file)
    args.min_samples = params['min_samples']
    args.eps = params['eps']
    args.allow_dis = params['distance']
    print(f"Using optimized parameters: eps {args.eps}, min_samples {args.min_samples}, allow_dis {args.allow_dis}")

    
    sdf_path = os.path.join(par_dir, args.sdf_t)
    suppl = Chem.SDMolSupplier(sdf_path)
    mols_train = [mol for mol in suppl]
    sdf_screen_path = args.sdf_s
    suppl = Chem.SDMolSupplier(sdf_screen_path)
    mols_screen = [mol for mol in suppl]
    print(f"Training molecules {len(mols_train)} Screening molecules {len(mols_screen)} ")
    # Need to align mols_screen
    # ------------------ #
    reference_dir = os.path.join(par_dir, "data/processed/references.sdf")

    suppl = Chem.SDMolSupplier(reference_dir)

    references = [mol for mol in suppl]
    references_dict = {}

    for mol in references:
        name = f"ref{mol.GetProp('reference')}"
        mol.SetProp("_Name", name)       
        references_dict[name] = mol      

    #Reference Indole
    # NOTE: This alignment somehow ruins the models performance!
    #p = Chem.MolFromSmiles('[nH]1ccc2ccccc21')
    #AllChem.Compute2DCoords(p)
    #conf_indole = p.GetConformer()
    #for i in range(p.GetNumAtoms()):
    #    pos = conf_indole.GetAtomPosition(i)
    #    conf_indole.SetAtomPosition(i, (-pos.x, -pos.y, pos.z))  # Flip horizontally
    #mols_screen = [mol for mol in mols_screen if mol.HasSubstructMatch(p)]
    #make_the_alignments(mols_screen, references_dict, p)
    #print("Aligned Molecules")
    #
    #---------#

    ic50s_con = [float(m.GetProp('IC50')) if m.GetProp('IC50') != '>100' else 100 for m in mols_train]
    ic50s = [1 if ic50 < 30 else 0 for ic50 in ic50s_con]
    
    centroids_path = os.path.join(par_dir, "results/models/centroids.npy")
    centroids = np.load(centroids_path)
    #TODO: Check why X_train here is not the same as the other!!!
    print(args.eps, args.allow_dis, args.min_samples)
    X_train, X_screen = make_descriptors_fast(mols_train, min_samples = args.min_samples,
                                                            eps = args.eps, allow_dis = args.allow_dis, mols_screen = mols_screen,
                                                            incl_screen_mols= True,
                                                            pre_calc_centroid=True, centroids=centroids)
    
    print("Centroids CHECK", np.sum(centroids), "X_train CHECK", np.sum(X_train))

    print(f"Succesfully made descriptors for X_train {np.shape(X_train)} and X_screen {np.shape(X_screen)}")
    path_tree = os.path.join(par_dir, "results/models/decision_tree.pkl")
    with open(path_tree, 'rb') as f:
        tree = pickle.load(f)
    
    pred = tree.predict(X_screen)
    print("Predicted")

    mol_pred = [m for m, p in zip(mols_screen, pred) if p == 1]
    x_pred = [x for x, p in zip(X_screen, pred) if p == 1]
    hit_dir = os.path.join(par_dir, "results/screening")
    os.makedirs(hit_dir, exist_ok=True)

    X_hits = []
    X_t = []

    writer = Chem.SDWriter(os.path.join(hit_dir, "hits.sdf"))
    writer_train = Chem.SDWriter(os.path.join(hit_dir, "train.sdf"))
    writer_full_screen = Chem.SDWriter(os.path.join(hit_dir, "screen.sdf"))


    for mol, x_h in zip(mol_pred, x_pred):
        mol.SetProp("hit", "y")
        writer.write(mol)
        X_hits.append(x_h)

    for mol_t, x_t in zip(mols_train, X_train):
        writer_train.write(mol_t)
        X_t.append(x_t)

    for mol_s in mols_screen:
        writer_full_screen.write(mol_s)

    writer.close()
    writer_train.close()
    writer_full_screen.close()

    np.save(os.path.join(hit_dir, "X_hits.npy"), np.array(X_hits))
    np.save(os.path.join(hit_dir, "X_train.npy"), np.array(X_t))
    np.save(os.path.join(hit_dir, "X_screen.npy"), np.array(X_screen))
if __name__ == "__main__":
    main()