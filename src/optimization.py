from loader import resolve_repo
resolve_repo()
import os
from descriptors import make_descriptors_fast
from models import train
from rdkit import Chem
import argparse
import numpy as np
from parameter_optimization import leave_one_out_cv
import optuna
from sklearn.metrics import matthews_corrcoef, confusion_matrix
import yaml
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import matthews_corrcoef, precision_score
import numpy as np
from sklearn.tree import DecisionTreeClassifier
def main():
    script_dir = os.getcwd()
    par_dir = os.path.dirname(script_dir)
    sdf_path = os.path.join(par_dir, "data/toy_indoles_aligned.sdf")
    res_path = os.path.join(par_dir, "results/models/params.yaml")
    parser = argparse.ArgumentParser(description=
                                     f"Please input your desired parameter boundaries for"
                                     "the parameter optimization!")
    parser.add_argument("--n_trials_n_startup",
                        type = float,
                        nargs=2,
                        metavar=("n_trials","n_startup"),
                        default=(100,10),
                        help = "Number of trials and number of warmup trials for TPEsampler optimization")
    
    parser.add_argument("--eps_range",
                        type = float,
                        nargs=2,
                        metavar=("MIN","MAX"),
                        default=(0.4,1),
                        help = "Range for eps in optimization. \n eps is the clustering metric that defines the distance limit to include points in the same neighbourhood. \n This is also used in the model as the radial distance within which atoms contribute fully to the descriptor.")
    
    parser.add_argument("--min_samples_range",
                        type = int,
                        nargs=2,
                        metavar=("MIN","MAX"),
                        default=(1,4),
                        help = "Integer range for min_samples in optimization. \n min_samples refers the minimum number of samples required to create a centroid")
    
    parser.add_argument("--allow_dis_range",
                        type = float,
                        nargs=2,
                        metavar=("MIN","MAX"),
                        default=(1,5),
                        help = "Range for weighting distance cutoff (allow_dis) in optimization. \n In the model, if atoms are beyond allow_dis from a centroid, they do not contribute to the properties of that centroid.")
    parser.add_argument("--sdf", type=str, default=sdf_path,
                        help="Path to input SDF file. Must include float IC50 value in uM.")
    
    args = parser.parse_args()
    if not any([args.n_trials_n_startup != (100,10), args.eps_range != (0.7, 1), args.min_samples_range != (1,4), args.allow_dis_range != (1,5)]):
        print("WARNING: You are using default parameters (eps_range 0.4-1, min_samples 1-5, allow_dis_range 1-5).")
        print("Set your own with --eps_range, --min_samples_range, --allow_dis_range.")
        print("Example: python train.py --eps_range 0.7 1.0 --min_samples_range 1 10 --allow_dis_range 1.2 1.5")
    
    n_trials = args.n_trials_n_startup[0]; n_warmup = args.n_trials_n_startup[1]
    eps_min = args.eps_range[0]; eps_max = args.eps_range[1]
    min_samples_min = args.min_samples_range[0]; min_samples_max = args.min_samples_range[1]
    allow_dis_min = args.allow_dis_range[0]; allow_dis_max = args.allow_dis_range[1]





    sdf_path = os.path.join(par_dir, args.sdf)
    suppl = Chem.SDMolSupplier(sdf_path)
    mols_train = [mol for mol in suppl]
    ic50s_con = [float(m.GetProp('IC50')) if m.GetProp('IC50') != '>100' else 100 for m in mols_train]
    ic50s = [1 if ic50 < 30 else 0 for ic50 in ic50s_con]
    ic50s = np.array(ic50s).flatten()

    def objective(trial):
        min_samples = trial.suggest_int("min_samples", min_samples_min, min_samples_max)          # integer
        eps = trial.suggest_float("eps", eps_min, eps_max)                     # continuous
        distance = trial.suggest_float("distance", allow_dis_min, allow_dis_max) 
        X_train = make_descriptors_fast(mols_train, min_samples, eps, distance, return_centroids=False, verbose = False)
        true_IC50, pred_IC50,  mols_list, trees_ = leave_one_out_cv(mols_train, ic50s, X_train)
        score = matthews_corrcoef(true_IC50, pred_IC50)
        return score


    sampler = optuna.samplers.TPESampler(n_startup_trials=n_warmup)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    study.optimize(objective, n_trials=n_trials)  # total trials (including random start)

    print("Best parameters:", study.best_params)
    print("Best score:", study.best_value)  #
    
    with open(res_path, "w") as f:
        yaml.dump(study.best_params, f, default_flow_style=False)

    print(f"Saved best parameters to {res_path}")

if __name__ == "__main__":
    main()