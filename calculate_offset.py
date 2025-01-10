import pandas as pd 
import numpy as np
import pdb

df_322 = pd.read_csv('WASP80b_transmission_NIRCam322.dat', delim_whitespace=True, skiprows=1, names=['wavelength', 'bin_width', 'transit depth', '+1sigma'])

#df_444 = pd.read_csv('WASP80b_transmission_NIRCam444.dat', delim_whitespace=True, skiprows=1, names=['wavelength', 'bin_width', 'transit depth', '+1sigma'])
df_444 = pd.read_csv('./transmission_spectra/jwst/NIRCam444/transmission_spectra_NIRCam444_gaussian_spot.txt', delim_whitespace=True, skiprows=1, names=['transit depth','+1sigma','err2','wavelength','bin_width'])

index_322 = np.where(df_322['wavelength'] > df_444['wavelength'][0])[0]
index_444 = np.where(df_444['wavelength'] < df_322['wavelength'][len(df_322['wavelength']) - 1])[0]

error_322 = df_322['+1sigma'][index_322]*1e6

error_444 = df_444['+1sigma'][index_444]*1e6

weights_322 = 1/(error_322**2.)
depth_322 = df_322['transit depth'][index_322]*1e6
avgdepth_322 = np.array(np.average(depth_322, weights=weights_322))

weights_444 = 1/(error_444**2.)
depth_444 = df_444['transit depth'][index_444]*1e6
avgdepth_444 = np.array(np.average(depth_444, weights=weights_444))

offset1 = (avgdepth_444 - avgdepth_322)

w_err = np.sqrt((1/np.sum(np.concatenate([1/(df_444['+1sigma'][index_444])**2, (1/df_322['+1sigma'][index_322])**2]))))*1e6
