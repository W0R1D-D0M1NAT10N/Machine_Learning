import pandas as pd
df = pd.read_csv(r"C:\Users\alexa\airfoil_data.csv")
df = df.drop_duplicates(subset=['file_path', 'aoa', 'cl'])  # Removes exact row dupes
df = df.sort_values(['file_path', 'aoa'])  # Orders for consistency
df.to_csv(r"C:\Users\alexa\airfoil_data_clean.csv", index=False)