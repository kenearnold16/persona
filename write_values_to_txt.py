import pandas as pd

def write_values_to_txt(priors, filename):

    df = pd.DataFrame.from_dict(priors)       
    df.to_csv(filename, sep='\t', index=True, na_rep="nan")
    
    return df
        
