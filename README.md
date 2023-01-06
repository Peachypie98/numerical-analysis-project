# numerical-analysis-project
## Aim
.

## Procedure
### Dataset
```shell
# allowable multiple choice node and edge features
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)) + ['misc'],
    'possible_bond_type_list' : [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
}

def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1

def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    atom_feature = [
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            ]
    return atom_feature

def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    bond_feature = [
                safe_index(allowable_features['possible_bond_type_list'], str(bond.GetBondType())),
            ]
    return bond_feature

def from_mol(mol_file, y=None, smiles=None):
    r"""Converts a Molecule data to a :class:`torch_geometric.data.Data`
    instance.

    Args:
        y(list, optional): target value
        mol_file (string, optional): The Mol filename string.
    """
    RDLogger.DisableLog('rdApp.*')
    
    if y is not None:
        y = torch.tensor(y, dtype=torch.float)
    mol = Chem.SDMolSupplier(mol_file, removeHs=False,
                                   sanitize=False)
    mol = mol[0]
    xs = []
    xc = []
    for i, atom in enumerate(mol.GetAtoms()):
        positions = mol.GetConformer().GetAtomPosition(i)
        x = atom_to_feature_vector(atom)
        xc.append([positions.x, positions.y, positions.z])
        xs.append(x)
    
    x = torch.tensor(xs, dtype=torch.float).view(-1, 1)  # Embedding 사용 시 dtype=torch.long
    xc = torch.tensor(xc, dtype=torch.float).view(-1, 3)
    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = bond_to_feature_vector(bond)

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float).view(-1, 1)  # Embedding 사용 시 dtype=torch.long

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x, pos=xc, y=y, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
```
