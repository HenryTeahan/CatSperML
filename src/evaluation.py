from scipy.linalg import norm
from descriptors import get_coords
from models import get_feature_name, get_test_path, get_feature_property, retrieve_molecules_from_node
from sklearn.tree import _tree
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.core.display import display_svg
import re
from rdkit.Geometry import Point2D, Point3D
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from scipy.linalg import norm
import os



def custom_metric(decision_tree, centroids, X_test, X_mol, eps):
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

def visualize_molecule_decision_path(final_tree, test_object, molecule, descriptors, centroids, eps, feature_names=None, numb_atom=1):
    path_info = get_test_path(final_tree, test_object, feature_names)
    svgs = []

    def edit_molecule(m, centroid_position):
        """ Adds a Neon (Ne) atom at the specified centroid position. """
        rwMol = Chem.RWMol(m)
        a = Chem.Atom(10)  # Neon atom
        centroid_idx = rwMol.AddAtom(a)

        if rwMol.GetNumConformers() == 0:
            rwMol.AddConformer(Chem.Conformer(rwMol.GetNumAtoms()))        

        pos = Point3D(float(centroid_position[0]), float(centroid_position[1]), 0.0)
        rwMol.GetConformer().SetAtomPosition(centroid_idx, pos)

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
                feature_value_text = f"No atom! Defaulting to {get_feature_name(feature)}={descriptor_value:.3f}"
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
# ----------------------
# Replacing old visualize function with new one - now uses inverted colour scheme.
# ----------------------
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
#
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
    
    

    size = size
    d2d = Draw.MolDraw2DCairo(*size)
    dopts = d2d.drawOptions()
    dopts.useBWAtomPalette()
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

    def draw_X(center, d2d, color=(0,0,0,1.0), size=0.6):
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
            draw_X(center, d2d, color=(1,0,0,0.4), size=0.5)

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
            draw_X(center, d2d, color=(1,0,0,0.65), size=0.5)

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
    images = [visualize(decision_tree, X_train[i], mols_train[i], centroids, eps, allow_dis, textanot=True, tex = f"index {i}, ic50: {float(alt_true_IC50[i]):.4}", textsize=10, size = (800,800)) for i in train_indices[:imagelen]]
    

    plot_image_grid(images, cols = col_leng, dpi = 300)

#def build_tree_graph_labeled(decision_tree, mols_train, X_train, mols_screen, X_screen, IC50, alt_true_IC50,
#                             density_scaling_factor=2000, color_intensity_factor=1.0):
#    tree_ = decision_tree.tree_
#    
#    decision_paths_train = decision_tree.decision_path(X_train).toarray()
#    decision_paths_screen = decision_tree.decision_path(X_screen).toarray()
#    print(density_scaling_factor)
#    if not isinstance(density_scaling_factor, (int, float)):
#        raise TypeError(f"density_scaling_factor must be a number, but got {type(density_scaling_factor)}")
#
#    node_frequencies_train = np.zeros(tree_.node_count, dtype=int)
#    node_frequencies_screen = np.zeros(tree_.node_count, dtype=int)
#
#    for sample in decision_paths_train:
#        node_frequencies_train += sample
#    for sample in decision_paths_screen:
#        node_frequencies_screen += sample
#
#    max_freq = max(max(node_frequencies_train), max(node_frequencies_screen))
#    if max_freq == 0:
#        max_freq = 1  # Avoid division by zero
#
#    visited_nodes = {node_id for node_id in range(tree_.node_count) if node_frequencies_screen[node_id] > 0}
#
#    if not visited_nodes:
#        raise ValueError("No nodes were visited by X_screen molecules.")
#
#    G = nx.DiGraph()
#    node_sizes = {}
#    node_colors = {}
#    node_labels = {}
#    node_class = {}
#
#    for node_id in visited_nodes:
#        left_child = tree_.children_left[node_id]
#        right_child = tree_.children_right[node_id]
#        if left_child and right_child == -1: # If active!
#            mols_train_node, _, _, _, temp_IC50,_,_ = retrieve_molecules_from_node(decision_tree, X_train, X_screen, mols_train, mols_screen, node_id, IC50, alt_true_IC50)
#
#            node_class[node_id] = temp_IC50[0]
#
#        else:
#            node_class[node_id] = -1
#
#        if left_child in visited_nodes:
#            G.add_edge(node_id, left_child)
#        if right_child in visited_nodes:
#            G.add_edge(node_id, right_child)
#
#    density_scaling_factor = 3000
#    for node_id in visited_nodes:
#        G.add_node(node_id)
#
#        visit_count_train = node_frequencies_train[node_id]
#        visit_count_screen = node_frequencies_screen[node_id]
#
#        node_sizes[node_id] = 500 + (density_scaling_factor * ((visit_count_train + visit_count_screen) / max_freq))
#
#        node_colors[node_id] = ((visit_count_train + visit_count_screen) / max_freq) ** color_intensity_factor
#
#
#        def get_feature_name(feature_index):
#            feature_map = {0: "logP", 1: "HBA", 2: "HBD", 3: "Atom"}  
#            property_type, centroid_index = get_feature_property(feature_index)
#
#            feature_name = feature_map.get(property_type, "Unknown")  
#            return feature_name, centroid_index
#        feature_index = tree_.feature[node_id]
#        #print(node_labels)
#        #print(G.nodes)
#
#        if feature_index != _tree.TREE_UNDEFINED:
# 
#            feature_name,  centroid= get_feature_name(feature_index)
#            node_labels[node_id] = (f"Feature {feature_index}\n Centroid {centroid} \n{feature_name}\n" +
#                                    f"Train Visits: {visit_count_train} \n | Screen Visits: {visit_count_screen}\n" 
#                                    )
#        else:
#            node_labels[node_id] = (f"Leaf Node\nTrain Visits: {visit_count_train} \n | Screen Visits: {visit_count_screen}\nNode: {node_id}\n"+
#                                    f"Class: {'Active' if node_class[node_id] == 1 else 'Inactive'}"
#                              )
#
#
#    return G, node_sizes, node_colors, node_labels
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
        if left_child == -1 and right_child == -1:
            mols_train_node, _, _, _, temp_IC50,_,_, train_indices = retrieve_molecules_from_node(decision_tree, X_train, X_screen, mols_train, mols_screen, node_id, IC50, alt_true_IC50)
            if len(train_indices) > 0:
                temp_IC50 = [IC50[i] for i in train_indices]
                node_class[node_id] = temp_IC50[0]
            else:
                node_class[node_id] = -1  # or some default
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
    
    plt.figure(figsize=(40, 20), dpi=500)
    nx.draw(G, pos, with_labels=False, node_size=[node_sizes[n] for n in G.nodes], 
            node_color=color_values, edge_color="gray", linewidths=1, cmap=cmap, alpha=0.7)

    label_pos = {node: (pos[node][0] + label_offset[0], pos[node][1] + label_offset[1]) for node in G.nodes}

    for node, (x, y) in label_pos.items():

        plt.text(x, y, node_labels[node], fontsize=15, fontweight="bold", color=text_color,
                 bbox=dict(facecolor=text_bg_color, alpha=text_bg_alpha, edgecolor='none', boxstyle="round,pad=0.3"))

    plt.title(f"Decision Tree Traversal by X_screen Molecules Using DBscan, min_samples = {min_samples}, eps = {eps}")
    # plt.savefig('/home/henry-teahan/Downloads/decision_tree.png', dpi=500)
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
   #return temp_df
#def get_data(proper, value, pos, choose_prop):
#    prop_df = pd.DataFrame()
#    prop_df['x'] = [pos[i][0] for i in range(len(pos))]
#    prop_df['y'] = [pos[i][1] for i in range(len(pos))]
#    prop_df['value'] = value
#    prop_df['property'] = proper
#   # print(len(cont_df))
#    prop_df = prop_df[prop_df['property']==choose_prop]
#    prop_df.sort_values(by='x', inplace=True)
#    mean_rows = []
#    rows = []
#    current_x = None
#    current_y = None
#
#    for i, row in prop_df.iterrows():
#        if row['x'] != current_x:
#            if rows:  
#                mean_value = np.mean(rows)
#                mean_rows.append({'x': current_x, 'y': current_y, 'mean_value': mean_value, 'rows': rows})
#            rows = [row['value']]
#            current_x = row['x']
#            current_y = row['y'] 
#        else:
#            rows.append(row['value'])
#    if rows:
#        mean_value = np.mean(rows)
#        mean_rows.append({'x': current_x, 'y': current_y, 'mean_value': mean_value, 'rows': rows})
#    rows = pd.DataFrame(mean_rows)
#    return rows

def train_and_predict_100(i, IC50, mols_train, X_train, mols_screen, X_screen, centroids, eps):

    final_tree_100 = DecisionTreeClassifier(max_depth=None, random_state=i)
    final_tree_100.fit(X_train, IC50)
    preds = final_tree_100.predict(X_screen)
    metrics = []
    for i in range(len(mols_screen)):
        m = custom_metric(final_tree_100, centroids, np.array(X_screen[i]), mols_screen[i], eps)
        metrics.append(m)

    return preds, final_tree_100, metrics

def custom_metric(decision_tree, centroids, X_test, X_mol, eps):
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

def plt_landscape(idx, X_idx, centroids, tree, n_x, n_y, IC50 = None, screen = False):
    desc = X_idx[idx]
    n_centroids = len(centroids)
    desc_reshaped = desc.reshape(n_centroids, 4)
    channels = ['logP', 'HBA', 'HBD']
    cmap_list = ['coolwarm', 'YlGnBu', 'Oranges']
    # Add different label for the centroids that are actually hit by the decision tree... Using get_test_path
    if (n_x + n_y)  == 4:
        fig, axes = plt.subplots(n_x, n_y, figsize=(n_y * 5, n_x * 5))
    else:
        raise ValueError("dimension must sum to 4 (3x1, 1x3)")
#    if screen == False:
        # need to use feature_name, i.e. the output needs to go on the plot where it is needed! so if i == 0: logP -> logP plot.
        #property_type, centroid_index = get_feature_property(int(328))
        # centroid id : i in centroids, property_type: 0-4 logP; hba, hbd, exist
            # Flatten axes if it's 2D (n_x > 1 and n_y > 1)
    if isinstance(axes, np.ndarray) and axes.ndim > 1:
        axes = axes.flatten()
    path = get_test_path(tree, X_idx[idx])
    path = np.array(path)
    cent_path = [int(path[:,0][i]) for i in range(len(path[:,0]))]
    logP = []; HBA = []; HBD = []; exists = []
    for point in cent_path:
        property_type, centroids_index = get_feature_property(point)
        if property_type == 0:
            logP.append(centroids_index)
        if property_type == 1:
            HBA.append(centroids_index)
        if property_type == 2:
            HBD.append(centroids_index)
    vlims = [(-0.5, 1), (0, 1), (0, 1)]  

    for i, ax in enumerate(axes):
        sc = ax.scatter(
            centroids[:,0], centroids[:,1],
            c=desc_reshaped[:, i],
            s=100,
            cmap=cmap_list[i],
            edgecolor='k',
            vmin=vlims[i][0],
            vmax=vlims[i][1]
        )
        ax.set_title(channels[i])
        ax.set_xlabel('X (Å)')
        if i == 0:
            ax.set_ylabel('Y (Å)')
        if i == 0:
            for k in logP:
                ax.scatter(centroids[k,0], centroids[k,1], s = 200, edgecolors = 'black', marker = "o", facecolors = 'None', linewidths = 3)       
        if i == 1:
            for k in HBA:
                ax.scatter(centroids[k,0], centroids[k,1], s = 200, edgecolors = 'black', marker = "o", facecolors = 'None', linewidths = 3)      
        if i == 2:
            for k in HBD:
                ax.scatter(centroids[k,0], centroids[k,1], s = 200, edgecolors = 'black', marker = "o", facecolors = 'None', linewidths = 3)   
        
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    if screen == True:
        plt.suptitle(f'Centroid Descriptor Landscapes for Screen Molecule {idx}', fontsize=16)
    else:
        if IC50:
            plt.suptitle(f'Centroid Descriptor Landscapes for Train Molecule {idx} IC50: {IC50[idx]:.3f} uM', fontsize=13)
        else:
            plt.suptitle(f'Centroid Descriptor Landscapes for Train Molecule {idx}', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plt_average_landscape(train_idx, X_train, centroids, n_x, n_y, IC50=None):
    n_centroids = len(centroids)
    lis_desc = []
    for idx in train_idx:
        desc = X_train[idx]
        desc_reshaped = desc.reshape(n_centroids, 4)
        lis_desc.append(desc_reshaped)
    desc_averaged = np.average(lis_desc, axis = 0)

    desc_averaged.reshape(n_centroids,4)
    channels = ['logP', 'HBA', 'HBD']
    cmap_list = ['coolwarm', 'YlGnBu', 'Oranges']
    vlims = [(-0.5, 1), (0, 1), (0, 1)]  
    if (n_x + n_y) == 4:
        fig, axes = plt.subplots(n_x, n_y, figsize=(n_y * 5, n_x * 5))

    else:
        raise ValueError("dimension must sum to 4 (3x1 or 1x3)")
    for i, ax in enumerate(axes):
        sc = ax.scatter(
            centroids[:,0], centroids[:,1],
            c=desc_averaged[:, i],
            s=100,
            cmap=cmap_list[i],
            edgecolor='k',
            vmin=vlims[i][0],
            vmax=vlims[i][1]
        )
        ax.set_title(channels[i])
        ax.set_xlabel('X (Å)')
#        if i == 0:
        ax.set_ylabel('Y (Å)')
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    else:
        if IC50:
            plt.suptitle(f'Average Centroid Descriptor Landscapes for Train Molecules {[idx for idx in train_idx]} IC50: {[IC50[idx] for idx in train_idx]} uM', fontsize=13)
        else:
            plt.suptitle(f'Average Centroid Descriptor Landscapes for Train Molecules {[idx for idx in train_idx]}', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def compare_landscape(train_idx, screen_idx, X_train, X_screen, centroids, IC50, tree, average_train = False, n_x = 3, n_y = 1):
    if average_train:
        plt_average_landscape(train_idx, X_train, centroids, n_x, n_y, IC50 = IC50)
    else:
        if type(train_idx) == int:
            plt_landscape(train_idx, X_train, centroids, tree, n_x, n_y, IC50 = IC50)
        elif isinstance(train_idx, list):
            for id in train_idx:
                plt_landscape(id, X_train, centroids, tree, n_x, n_y, IC50 = IC50)

    if type(screen_idx) == int:
        plt_landscape(screen_idx, X_screen, centroids, tree, n_x, n_y, screen = True)
    elif isinstance(screen_idx, list):
        for id in screen_idx:
            plt_landscape(id, X_screen, centroids, tree, n_x, n_y, screen = True)