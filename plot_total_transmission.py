import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pdb

fontsize=25
plt.figure(1)
plt.figure(figsize=(18, 8))


filename = './transmission_spectra/hst/transmission_spectra_hst.txt'
df_hst = pd.read_csv(filename, delim_whitespace=True, skiprows=1, names=['transit depth', '+1sigma', '-1sigma', 'wavelength', 'bin_width'])

CF = pd.read_csv('correction_factor_hst.txt',  delim_whitespace=True)

plt.errorbar(df_hst['wavelength'], (df_hst['transit depth']*1e6), yerr=[df_hst['+1sigma']*1e6, df_hst['-1sigma']*1e6], label='PACMAN - mps2', c='forestgreen', fmt='o', alpha=0.8, zorder=10)

###############################################################################
###############################################################################

filename = './transmission_spectra/hst/transmission_spectra_hst_stagger.txt'
df_hst = pd.read_csv(filename, delim_whitespace=True, skiprows=1, names=['transit depth', '+1sigma', '-1sigma', 'wavelength', 'bin_width'])

CF = pd.read_csv('correction_factor_hst.txt',  delim_whitespace=True)

#plt.errorbar(df_hst['wavelength'], (df_hst['transit depth']*1e6), yerr=[df_hst['+1sigma']*1e6, df_hst['-1sigma']*1e6], label='PACMAN - stagger3D', c='blue', fmt='o', alpha=0.8, zorder=10)

###############################################################################
###############################################################################

filename = './transmission_spectra/hst/transmission_spectra_hst_previous.txt'

df_hst = pd.read_csv(filename, sep='\t', header=[0])

hst_depth = (df_hst['radius']**2)*1e6 
sigma_hst = np.sqrt((df_hst['radius']/2)*(df_hst['err'])**2)*1e6

plt.errorbar((df_hst['Wv']), hst_depth, yerr = (sigma_hst), c='red', fmt='o', alpha=1.0, label='Wong et al.')

###############################################################################
###############################################################################

#plt.ylim([29000, 30400])
plt.xlim([1.0, 1.8])

plt.xlabel(r'wavelength ($\mu$m)', fontsize=fontsize)
plt.ylabel('transit depth (ppm)', fontsize=fontsize)
plt.title('WASP80-b transmission spectra', fontsize=fontsize)

plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

plt.legend(loc='upper left', fontsize=20)

plt.savefig('./figs/total_transmission_spectra.png', dpi=300, bbox_inches='tight')

