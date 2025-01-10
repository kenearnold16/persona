import pandas as pd 
import numpy as np

filename = './transmission_spectra/jwst/NIRCam444/transmission_spectra_NIRCam444_spot_ignored.txt'

df = pd.read_csv(filename, delim_whitespace=True, skiprows=1, names = ['depth', 'err', '-1sigma', 'wv', 'bin_width'])

df['wv'] = np.round(df['wv'], 4)

redo = df.reindex(['wv', 'bin_width', 'depth', 'err'], axis=1)

redo.to_csv('WASP80b_transmission_NIRCam444_spot_ignored.dat', sep='\t', index=False)