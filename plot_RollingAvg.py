import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic, chi2, norm, anderson
#import seaborn as sns
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import pdb
import pandas as pd

def plot_data(ax, lam, td, error, n, label, color='green', name='Data', marker='.'):
    rolling=[]
    for i in range(n,len(lam)-n):
        rolling.append(np.mean(td[i-n:i+n+1]))

    ax.plot(lam[n:-n],np.asarray(rolling),color=color, alpha=1, lw=2, label=label, marker=marker, linestyle=None)
    data_bins=np.abs(lam[1:]-lam[:-1])/2.0
    data_bins=np.append(data_bins,data_bins[-1])
    #ax.scatter(lam, td,color=color,label=name)
    #ax.errorbar(lam, td, yerr=error, xerr=data_bins, marker=marker,markersize=6, elinewidth=2, capsize=3, capthick=1.2, ls='none', color=color,markeredgecolor=color, ecolor=color,label=name)

#sns.set_style('ticks')

fig=plt.figure(figsize=(10, 4))
spec = plt.gca()

n=3
#filename = './transmission_spectra/jwst/NIRCam322/transmission_spectra_NIRCam322.txt'
filename = 'WASP80b_transmission_NIRCam322.dat'
df_322 = pd.read_csv(filename, delim_whitespace=True, skiprows=1, names=['wavelength', 'bin_width', 'transit depth', '+1sigma'])

depth_322=np.array(df_322['transit depth'])
error_322=np.array(df_322['+1sigma'])
wave_322=np.array(df_322['wavelength'])

depth_322 *= 1e6
error_322 *= 1e6

#depth = depth# - 79

###############################################################################
###############################################################################


#filename = './transmission_spectra/jwst/NIRCam444/transmission_spectra_NIRCam444.txt'
filename = 'WASP80b_transmission_NIRCam444.dat'
df_444 = pd.read_csv(filename, delim_whitespace=True, skiprows=1, names=['wavelength', 'bin_width', 'transit depth', '+1sigma'])

depth_444=np.array(df_444['transit depth'])
error_444=np.array(df_444['+1sigma'])
wave_444=np.array(df_444['wavelength'])

depth_444 *= 1e6
error_444 *= 1e6

med_depth_322 = np.median(depth_322)
med_depth_444 = np.median(depth_444)

label='322 unmasked without offset'

plot_data(spec, wave_322, depth_322, error_322, n, label, color='blue')

depth_322 = depth_322 - 130# - med_depth_322
depth_444 = depth_444# - med_depth_444

label='444 unmasked'

plot_data(spec, wave_444, depth_444, error_444, n, label, color='darkorchid')

label='322 unmasked with offset (-130 ppm)'

plot_data(spec, wave_322, depth_322, error_322, n, label, color='dodgerblue')

depth_322 = depth_322 + 130# - med_depth_322
depth_322 = depth_322 - 52# - med_depth_322
label='322 unmasked with offset (-52 ppm)'
plot_data(spec, wave_322, depth_322, error_322, n, label, color='navy')

df_444 = pd.read_csv('./transmission_spectra/jwst/NIRCam444/transmission_spectra_NIRCam444_spot_masked.txt', delim_whitespace=True, skiprows=1, names=['transit depth','+1sigma','err2','wavelength','bin_width'])

depth_444=np.array(df_444['transit depth'])*1e6
error_444=np.array(df_444['+1sigma'])
wave_444=np.array(df_444['wavelength'])

label='444 masked'

plot_data(spec, wave_444, depth_444, error_444, n, label, color='forestgreen')


###############################################################################
###############################################################################


#filename = 'WASP80b_F322W2_dl015_Bell_v1.dat'

#df_eureka322 = pd.read_csv(filename, skiprows=1, delim_whitespace=True, names=['wavelength', 'bin_width', 'depth', 'error'])
#depth_322=np.array(df_eureka322['depth']) 
#error_322=np.array(df_eureka322['error'])
#wave_322=np.array(df_eureka322['wavelength'])

#depth_322 *= 1e6
#error_322 *= 1e6

#depth = depth# - 159

###############################################################################
###############################################################################


#filename = 'WASP80b_F444W_dl015_Bell_v1p1.dat'

#df_eureka444 = pd.read_csv(filename, skiprows=1, delim_whitespace=True, names=['wavelength', 'bin_width', 'depth', 'error'])
#depth_444=np.array(df_eureka444['depth'])
#error_444=np.array(df_eureka444['error'])
#wave_444=np.array(df_eureka444['wavelength'])

#depth_444 *= 1e6
#error_444 *= 1e6

#med_depth_322 = np.median(depth_322)
#med_depth_444 = np.median(depth_444)

#depth_322 = depth_322 - med_depth_322

#depth_444 = depth_444 - med_depth_444

#label='eureka!'

#plot_data(spec, wave_322, depth_322, error_322, n, label, color='purple')

#label='_nolegend_'

#plot_data(spec, wave_444, depth_444, error_444, n, label, color='purple')

###############################################################################
###############################################################################


#filename = 'spectrum_lc_fit_wasp80b_f322w2_spec_trans_p010_joint_t_fixLD.csv'

#df_tshirt322 = pd.read_csv(filename, header=[0])
#depth_322=np.array(df_tshirt322['depth'])
#error_322=np.array(df_tshirt322['depth err'])
#wave_322=np.array(df_tshirt322['wave mid'])

#depth_322 *= 1e6
#error_322 *= 1e6

#depth = depth# - 8

###############################################################################
###############################################################################

#filename = 'spectrum_lc_fit_wasp80b_f444w_spec_trans_p003_joint_t_fixLD.csv'

#df_tshirt444 = pd.read_csv(filename, header=[0])
#depth_444=np.array(df_tshirt444['depth'])
#error_444=np.array(df_tshirt444['depth err'])
#wave_444=np.array(df_tshirt444['wave mid'])

#depth_444 *= 1e6
#error_322 *= 1e6

#med_depth_322 = np.median(depth_322)
#med_depth_444 = np.median(depth_444)

#depth_322 = depth_322 - med_depth_322

#depth_444 = depth_444 - med_depth_444

#label='_nolegend_'

#plot_data(spec, wave_444, depth_444, error_322, n, label, color='forestgreen')

#label='tshirt'

#plot_data(spec, wave_322, depth_322, error_322, n, label, color='forestgreen')

###############################################################################
###############################################################################


spec.xaxis.set_minor_locator(MultipleLocator(0.1))
spec.yaxis.set_minor_locator(MultipleLocator(20))

plt.grid()
plt.xlim(2.4,5.0)

plt.ylabel('depth', fontsize=14)
plt.xlabel('Wavelength ($\mu$m)', fontsize=14)

#sns.despine(offset=10, trim=True)
plt.legend(loc='lower left')

plt.subplots_adjust(top=0.98, right=0.95, left=0.08, bottom=0.16)

plt.savefig("Rolling_avg_masked_vs_unmasked.png", dpi=300)
