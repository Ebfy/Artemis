import pandas as pd
import os

block_start = 8000000
block_end = 8999999
data_dir = './dataset'

print("Vérification du contenu de chaque fichier...")
for filename in os.listdir(data_dir):
    if filename.startswith('8000000to8999999') and filename.endswith('.csv'):
        full_path = os.path.join(data_dir, filename)
        print(f'\n→ Lecture de {filename}...')
        df = pd.read_csv(full_path, low_memory=False)
        block_col = next((col for col in df.columns if 'block' in col.lower()), None)
        if block_col is None:
            print("  ❌ Colonne block non trouvée.")
            continue
        df_block = df[(df[block_col] >= block_start) & (df[block_col] <= block_end)]
        print(f"  ✔ Bloc détecté : {block_col}")
        print(f"  ➤ Nombre de transactions dans plage [{block_start}, {block_end}] : {len(df_block)}")
