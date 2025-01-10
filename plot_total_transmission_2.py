import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pdb
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import binned_statistic
from astropy.convolution import convolve, Gaussian1DKernel

spot_type = 'gaussian_spot'

fontsize=25
fig = plt.figure(figsize=[20,10])
axMain = plt.subplot()
axMain.set_xscale('linear')
axMain.set_xlim((2.4, 5.1))
axMain.set_ylim((28500, 30000))

axMain.spines['right'].set_visible(False)
axMain.yaxis.set_ticks_position('left')
axMain.yaxis.set_visible(True)
axMain.tick_params(axis='y', labelsize=fontsize)
axMain.set_xlabel(r'                                              wavelength ($\mu$m)', fontsize=fontsize)
axMain.set_ylabel('transit depth (ppm)', fontsize=fontsize)
axMain.set_xticks([2.5,3.0,3.5,4.0,4.5], labels = [2.5,3.0,3.5,4.0,4.5], fontsize=fontsize)

###############################################################################


divider = make_axes_locatable(axMain)
axMIRI = divider.append_axes("right", size=5.0, pad=0, sharey=axMain)
axMIRI.set_xscale('linear')
axMIRI.set_xlim((5.0, 11))
axMIRI.spines['left'].set_visible(False)
axMIRI.yaxis.set_ticks_position('right')
plt.setp(axMIRI.get_xticklabels(), visible=True)
axMIRI.set_xticks([5.0, 9.0, 11.0], labels = [5.0, 9.0, 11.0], fontsize=fontsize)
axMIRI.yaxis.set_visible(False)


###############################################################################

if spot_type == 'spot_masked':
    filename = 'WASP80b_transmission_NIRCam444_spot_masked.dat'
    axMain.set_title('WASP80-b transmission spectra spot masked', fontsize=fontsize)
elif spot_type == 'triangle_spot':
    filename = 'WASP80b_transmission_NIRCam444_triangle_spot.dat'
    axMain.set_title('WASP80-b transmission spectra triangle spot', fontsize=fontsize)
elif spot_type == 'gaussian_spot':
    filename = 'WASP80b_transmission_NIRCam444_gaussian_spot.dat'
    axMain.set_title('                                           WASP80-b transmission spectra', fontsize=fontsize)
elif spot_type == 'spot_ignored':
    filename = 'WASP80b_transmission_NIRCam444.dat'
    axMain.set_title('WASP80-b transmission spectra spot ignored', fontsize=fontsize)

df_444 = pd.read_csv(filename, delim_whitespace=True, skiprows=1, names=['wavelength', 'bin_width', 'transit depth', '+1sigma'])

filename = 'WASP80b_transmission_NIRCam322.dat'
df_322 = pd.read_csv(filename, delim_whitespace=True, skiprows=1, names=['wavelength', 'bin_width', 'transit depth', '+1sigma'])

###############################################################################

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

w_err1 = np.sqrt((1/np.sum(np.concatenate([1/(df_444['+1sigma'][index_444])**2, (1/df_322['+1sigma'][index_322])**2]))))*1e6

###############################################################################

df_322['transit depth'] = (df_322['transit depth'])+(offset1/1e6)

axMain.errorbar(df_322['wavelength'], (df_322['transit depth']*1e6), yerr=[df_322['+1sigma']*1e6, df_322['+1sigma']*1e6], label='NIRCam/F322W2', c='dodgerblue', fmt='o', alpha=0.8, zorder=20, mfc="white",)
axMain.errorbar(df_444['wavelength'], (df_444['transit depth']*1e6), yerr=[df_444['+1sigma']*1e6, df_444['+1sigma']*1e6], label='NIRCam/F444W', c='darkorchid', fmt='o', alpha=0.8, zorder=20, mfc="white",)

###############################################################################

filename = 'Bell_WASP80b_LRS_dl0125_Bell_v1p3a.dat'
df_MIRI = pd.read_csv(filename, delim_whitespace=True, skiprows=1, names=['wavelength', 'bin_width', 'transit depth', '+1sigma'])

###############################################################################
index_444 = np.where(df_444['wavelength'] > df_MIRI['wavelength'][0])[0]
index_MIRI = np.where(df_MIRI['wavelength'] < df_444['wavelength'][len(df_444['wavelength']) - 1])[0]

error_MIRI = df_MIRI['+1sigma'][index_MIRI]*1e6

error_444 = df_444['+1sigma'][index_444]*1e6

#weights_MIRI = 1/(error_MIRI**2.)
#depth_MIRI = df_MIRI['transit depth'][index_MIRI]*1e6
#avgdepth_MIRI = np.array(np.average(depth_MIRI, weights=weights_MIRI))

#weights_444 = 1/(error_444**2.)
#depth_444 = df_444['transit depth'][index_444]*1e6
#avgdepth_444 = np.array(np.average(depth_444, weights=weights_444))

#offset2 = (avgdepth_444 - avgdepth_MIRI)

#w_err2 = np.sqrt((1/np.sum(np.concatenate([1/(df_444['+1sigma'][index_444])**2, (1/df_MIRI['+1sigma'][index_MIRI])**2]))))*1e6

###############################################################################
axMain.errorbar(df_MIRI['wavelength'], (df_MIRI['transit depth']*1e6), yerr=[df_MIRI['+1sigma']*1e6, df_MIRI['+1sigma']*1e6], label='MIRI/LRS', c='orange', fmt='o', alpha=0.8, zorder=20, mfc="white",)
axMIRI.errorbar(df_MIRI['wavelength'], (df_MIRI['transit depth']*1e6), yerr=[df_MIRI['+1sigma']*1e6, df_MIRI['+1sigma']*1e6], label='MIRI/LRS', c='orange', fmt='o', alpha=0.8, zorder=20, mfc="white",)

axMain.legend(loc='lower left', fontsize=25)

#axMain.errorbar(df_322_spot_masked['wv'], (df_322_spot_masked['depth']*1e6)-130, yerr=[df_322_spot_masked['err1']*1e6, df_322_spot_masked['err2']*1e6], label='NIRCam/F322W spot masked', c='forestgreen', fmt='o', alpha=0.9, zorder=20, mfc="white",)
#axMain.errorbar(df_444_spot_masked['wv'], (df_444_spot_masked['depth']*1e6), yerr=[df_444_spot_masked['err1']*1e6, df_444_spot_masked['err2']*1e6], label='NIRCam/F444W spot masked', c='forestgreen', fmt='o', alpha=0.9, zorder=20, mfc="white",)

if spot_type == 'spot_masked':
    plt.savefig('WASP-80b_transmission_spot_masked.png', dpi=300)
elif spot_type == 'triangle_spot':
    plt.savefig('WASP-80b_transmission_triangle_spot.png', dpi=300)
elif spot_type == 'spot_ignored':
    plt.savefig('WASP-80b_transmission_spot_ignored.png', dpi=300)
elif spot_type == 'gaussian_spot':
    plt.savefig('./figs/WASP-80b_transmission.png', dpi=300)

df_322.to_csv('./WASP-80b transmission spectra/WASP80b_transmission_NIRCam322_offset.txt', sep='\t', index=False)
df_444.to_csv('./WASP-80b transmission spectra/WASP80b_transmission_NIRCam444_gaussian_spot.txt', sep='\t', index=False)
df_MIRI.to_csv('./WASP-80b transmission spectra/WASP80b_transmission_MIRI_LRS.txt', sep='\t', index=False)
