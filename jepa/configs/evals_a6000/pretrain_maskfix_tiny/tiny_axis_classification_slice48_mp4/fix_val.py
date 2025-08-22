import pandas as pd


#split path in path, if same string in path, remove 2nd
# eg  /media/backup_16TB/sean/Monai/BrainSlice48_224_224/SOOP/sub-38_FLAIR/sub-38_FLAIR/Axial.mp4 
# should return /media/backup_16TB/sean/Monai/BrainSlice48_224_224/SOOP/sub-38_FLAIR/Axial.mp4
def remove_duplicate_paths(df):
    """Remove duplicate segments in the file path."""
    def clean_path(path):
        parts = path.split('/')
        cleaned_parts = []
        for part in parts:
            # Add part to cleaned_parts only if it's not a duplicate of the last added part
            if not cleaned_parts or cleaned_parts[-1] != part:
                cleaned_parts.append(part)
        return '/'.join(cleaned_parts)
    
    df['path'] = df['path'].apply(clean_path)
    return df.drop_duplicates(subset='path')

df = pd.read_csv('VJEPA/jepa/configs/evals_a6000/tiny_axis_classification_slice48_mp4/Brainslice_48_Axisclassification_val.csv',
                    sep=' ', header=None, names=['path', 'dummy'])
df = remove_duplicate_paths(df)
df.to_csv('VJEPA/jepa/configs/evals_a6000/tiny_axis_classification_slice48_mp4/Brainslice_48_Axisclassification_val_cleaned.csv',
          index=False, header=False, sep=' ')



