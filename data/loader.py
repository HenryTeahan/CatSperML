
import rdkit
from rdkit import Chem

def get_indole(pos_ref = False):
    p = Chem.MolFromSmiles('[nH]1ccc2ccccc21')
    p.Compute2DCoords()
    pc = p.GetConformer()
    change_sign = False
    for i in range(p.GetNumAtoms()):
        if i == 0:
            if pc.GetAtomPosition(i)[1] > 0:
                q = -1
            else:
                q = 1
        pos = pc.GetAtomPosition(i)
        # Rotate by 180° around the origin: (x, y) -> (-x, -y)
        pc.SetAtomPosition(i, (-pos.x, q*pos.y, pos.z))
    ref_indol = [pc.GetAtomPosition(i) for i in range(p.GetNumAtoms())]
    if pos_ref:
        return p, ref_indol
    else:
        return p
