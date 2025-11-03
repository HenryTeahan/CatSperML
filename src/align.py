
import argparse
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from loader import resolve_repo
import numpy as np
resolve_repo()

from data.loader import get_indole
import os
def reorient(mol, p, ref_indol, conf=None, force_reconf=False):
    """
    Reorient mol by aligning its indole substructure to ref_indol.

    - mol: RDKit Mol object
    - p: Indole scaffold
    - ref_indol: List of RDKit Point3D reference coordinates
    - conf: RDKit Conformer object (must match mol)
    """
    if conf is None:
        conf = mol.GetConformer()
    if force_reconf:
        mol.Compute2DCoords()
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

def make_the_alignments(mols, ref_dir, p):
    '''
    Function that applies priority-based realignment -> Largest fragment first etc.
    '''
    refs = {}
    for mol in ref_dir:

        ref_id = mol.GetProp("reference")
        refs[ref_id] = mol
    print(refs)
    print(len(refs.keys()))
    for i, key in enumerate(refs.keys(), start=1):
        
        print(key)
        
        globals()[f"reference{i}"] = ref_dir[int(key)-1]
    
    for i, m in enumerate(mols):
        if m.HasSubstructMatch(reference2): 
          print('has ref2')
          _= AllChem.GenerateDepictionMatching2DStructure(m,reference2)
          continue
        if m.HasSubstructMatch(reference10):
          print('has ref10')
          _ = AllChem.GenerateDepictionMatching2DStructure(m,reference10)
          continue
        if m.HasSubstructMatch(reference5):
          print('has ref 5')
          _=AllChem.GenerateDepictionMatching2DStructure(m,reference5)
          continue
        if m.HasSubstructMatch(reference14):
          print('has ref 14')
          _=AllChem.GenerateDepictionMatching2DStructure(m,reference14)
          continue
        if m.HasSubstructMatch(reference15):
          print('has ref15')
          _ = AllChem.GenerateDepictionMatching2DStructure(m, reference15)
          continue
        if m.HasSubstructMatch(reference13):
          print('has ref13')
          _ = AllChem.GenerateDepictionMatching2DStructure(m,reference13)
          continue
        if m.HasSubstructMatch(reference8):
          print('has ref8')
          _ = AllChem.GenerateDepictionMatching2DStructure(m,reference8)
          continue
        if m.HasSubstructMatch(reference16):
          _ = AllChem.GenerateDepictionMatching2DStructure(m,reference16)
          print('has ref16')
          continue
        if m.HasSubstructMatch(reference17):
          _ = AllChem.GenerateDepictionMatching2DStructure(m,reference17)

          print('has ref17')
          continue
        if m.HasSubstructMatch(reference6):
          print('has ref6')
          _ = AllChem.GenerateDepictionMatching2DStructure(m,reference6)
          continue
        if m.HasSubstructMatch(reference3): #Methoxy scaffold
          print('has ref 3') 
          _= AllChem.GenerateDepictionMatching2DStructure(m,reference3)
          #metoxy.append(copy(m))
          continue
        if m.HasSubstructMatch(reference7):
          print('has ref7')
          _ = AllChem.GenerateDepictionMatching2DStructure(m,reference7)
          continue
        if m.HasSubstructMatch(reference9):
          print('has ref9')
          _ = AllChem.GenerateDepictionMatching2DStructure(m,reference9)
          continue
        if m.HasSubstructMatch(reference11):
          print('has ref11')
          _ = AllChem.GenerateDepictionMatching2DStructure(m,reference11)

          continue
        if m.HasSubstructMatch(reference12):
          print('has ref12')
          _ = AllChem.GenerateDepictionMatching2DStructure(m,reference12)
          continue
        if m.HasSubstructMatch(reference4):
          print('has ref 4')
          _= AllChem.GenerateDepictionMatching2DStructure(m,reference4)
          continue
        if m.HasSubstructMatch(reference1): #Indole benzyl scaffold
          _ = AllChem.GenerateDepictionMatching2DStructure(m,reference1)
          print('has ref 1')
          continue

        else: #No scaffold
          print('no scaf')
          _ = AllChem.GenerateDepictionMatching2DStructure(m,get_indole())
          reorient(m, get_indole(), get_indole(pos_ref=True)[1], force_reconf = False)

def main():
    script_dir = os.getcwd()
    par_dir = os.path.dirname(script_dir)
    sdf_file = os.path.join(par_dir, "data/toy_indoles.sdf")
    parser = argparse.ArgumentParser(description="Indole aligner: input unaligned indoles SDF -> aligned indoles SDF")
    parser.add_argument("--sdf", type=str,
                    help="Path to input SDF file that contains un-aligned indoles", default = sdf_file)
    args = parser.parse_args()

    if args.sdf:
        try:
            reader = Chem.SDMolSupplier(args.sdf)
        except:
            raise ValueError("Not a proper SDF or directory")
        
        mols = [m for m in reader]
        print(f"loaded {len(mols)} molecules")
        print(get_indole())
        mols = [m for m in mols if m.HasSubstructMatch(get_indole())]
        print(f"... of which {len(mols)} indoles")
        ref_sdf = Chem.SDMolSupplier(os.path.join(par_dir, "data/processed/references.sdf"))
        ref = [m for m in ref_sdf]
        make_the_alignments(mols, ref, get_indole())
        newname = args.sdf[:-4] +"_aligned.sdf"
        writer = Chem.SDWriter(newname)
        for m in mols:
           writer.write(m)
        writer.close()
        print("saved aligned indoles to", newname)
    else:
        raise ValueError("No input")
    
if __name__ == "__main__":
    main()