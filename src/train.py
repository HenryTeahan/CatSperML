from rdkit import Chem
import pandas as pd
import os
from descriptors import make_descriptors_fast, draw_centroids
from models import train
from evaluation import visualize
import argparse
from rdkit.Geometry import Point2D
import yaml
import pickle 
import numpy as np 

def main():
    script_dir = os.getcwd()
    par_dir = os.path.dirname(script_dir)
    sdf_path = os.path.join(par_dir, "data/toy_indoles_aligned.sdf")
    ref_dir = os.path.join(par_dir, "data/processed/references.sdf")
    parser = argparse.ArgumentParser(description="Please input your desired parameters for the descriptor generation!")
    parser.add_argument("--eps", type=float, default=0.5, help="DBSCAN epsilon value")
    parser.add_argument("--min_samples", type=int, default=2, help="DBSCAN min_samples")
    parser.add_argument("--allow_dis", type=float, default=1.8, help="Allowed distance threshold")
    parser.add_argument("--sdf", type=str, default=sdf_path,
                        help="Path to input SDF file")
    parser.add_argument("--save_img", action="store_true", default = False, help = "Save visualization images of training molecules to results/Images")
    parser.add_argument("--load_opt_params", action="store_true", default = False, help = "Load parameters from optimization - optimization parameters are overwritten by default if not specified w. this command")
    args = parser.parse_args()


    if args.load_opt_params:
        param_path = os.path.join(par_dir, "results/models/params.yaml")
        with open(param_path, 'r') as file:
            params = yaml.safe_load(file)

        args.min_samples = params['min_samples']
        args.eps = params['eps']
        args.allow_dis = params['distance']
        print(f"Using optimized parameters: eps {args.eps}, min_samples {args.min_samples}, allow_dis {args.allow_dis}")
    if not any([args.eps
                 != 0.5, args.min_samples != 2, args.allow_dis != 1.8]):
        print("WARNING: You are using default parameters (eps=0.5, min_samples=2, allow_dis=1.8).")
        print("Set your own with --eps, --min_samples, --allow_dis.")
        print("Example: python train.py --eps 0.7 --min_samples 1 --allow_dis 2")
    param_path = os.path.join(par_dir, "results/models/params.yaml")
    
    params = {}
    params['min_samples'] = args.min_samples
    params['eps'] = args.eps
    params['distance'] = args.allow_dis
    with open(param_path, 'w') as file:
        params = yaml.safe_dump(params, file)

    sdf_path = args.sdf
    suppl = Chem.SDMolSupplier(sdf_path)
    mols_train = [mol for mol in suppl]
    ic50s_con = [float(m.GetProp('IC50')) if m.GetProp('IC50') != '>100' else 100 for m in mols_train]
    ic50s = [1 if ic50 < 30 else 0 for ic50 in ic50s_con]
    print(f'Loaded data - {len(mols_train)} training molecules')
    print(args.eps, args.allow_dis, args.min_samples)

    X_train, centroids = make_descriptors_fast(mols_train,
                                                  min_samples = args.min_samples, eps = args.eps, 
                                                  allow_dis = args.allow_dis, return_centroids=True)
    fig = draw_centroids(mols_train, min_samples = args.min_samples, eps = args.eps, draw=True)

    #Save centroids here!
    print("centroids CHECK",np.sum(centroids), "X_train CHECK", np.sum(X_train))
    centroids_path = os.path.join(par_dir, "results/models/centroids.npy")
    X_train_path = os.path.join(par_dir, "results/models/X_train_old.npy")
    np.save(centroids_path, centroids)
    np.save(X_train_path, X_train)
    final_tree = train(X_train, ic50s)

    print(f"Shape of X_train: {np.shape(X_train)}")
    tree_path = os.path.join(par_dir, "results", "models/decision_tree.pkl")
    with open(tree_path, "wb") as f:
        pickle.dump(final_tree, f)


    out_dir = os.path.join(par_dir, "results", "Images")
    os.makedirs(out_dir, exist_ok=True)
    # TODO: Test this draw functionality still works..
    if args.save_img:
        fig.savefig(os.path.join(out_dir, f"centroid_map.png"))
        for i, (mol, X) in enumerate(zip(mols_train, X_train)):
            img = visualize(
                final_tree, X, mol, centroids,
                eps=args.eps, allow_dis=args.allow_dis,
                tex=f"mol {i+1} - ic50 {ic50s_con[i]}", textanot=True, tex_pos=Point2D(0,2)
            )
            out_path = os.path.join(out_dir, f"result_{i+1:03}.png")
            img.save(out_path)
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()