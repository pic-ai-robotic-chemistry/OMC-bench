  from rdkit import Chem

  def to_canonical(smi: str):
      try:
          mol = Chem.MolFromSmiles(smi)
          if mol is None:
              return None
          return Chem.MolToSmiles(mol, canonical=True)
      except Exception:
          return None

  def build_smiles_stats(smiles_list):
      rows = []
      bad = []
      for smi in smiles_list:
          try:
              mol = Chem.MolFromSmiles(smi)
              if mol is None:
                  bad.append(smi)
                  continue
              heavy = mol.GetNumHeavyAtoms()
              total = Chem.AddHs(mol).GetNumAtoms()
              rows.append({'smiles': smi, 'heavy_atoms': heavy, 'total_atoms': total})
          except Exception:
              bad.append(smi)
      return rows, bad
