from exotic_ld import StellarLimbDarkening
import numpy as np
import pandas as pd
import pdb
# Path to store stellar and instrument data.
ld_data_path = 'exotic_ld_data'

# Stellar models grid.
ld_model = 'mps2'

# Metallicty [dex].
M_H = 0.12

# Effective temperature [K].
Teff = 3457

# Surface gravity [dex].
logg = 4.78

sld = StellarLimbDarkening(M_H, Teff, logg, ld_model, ld_data_path)

# Start and end of wavelength interval [angstroms].
#wv_min = 3.8975 #microns
#wv_max = 4.9625 #microns
#wv_min = 2.4575 #microns
#wv_max = 3.9425 #microns
#wavelength = np.arange(wv_min, wv_max+(0.015/2), 0.015)*10000
#wv_min = 3.8975
#wv_max = 4.9925

#wavelength = np.arange(wv_min, wv_max + (0.015/2), 0.015)*10000
wavelength = np.array([1.148, 1.173, 1.198, 1.224, 1.249, 1.274, 1.300, 1.325, 1.350, 1.376, 1.401, 1.427, 1.452, 1.477, 1.503, 1.528, 1.553, 1.579, 1.604, 1.629])*10000

#wavelength = np.array([np.median(wavelength)])

delta_wv = 250

mode = 'HST_WFC3_G141'

LD_type = 'quadratic'

filters = 'WFC3'

N = len(wavelength)

u1 = np.zeros(N)
u2 = np.zeros(N)
err_u1 = np.zeros(N)
err_u2 = np.zeros(N)

for i in range(-1, N - 1):
    i += 1
    try:
        wavelength_range = [int(wavelength[i]) - delta_wv, int(wavelength[i]) + delta_wv]
        
        if LD_type == 'quadratic':
                  (u1[i], u2[i]), (err_u1[i], err_u2[i]) = sld.compute_quadratic_ld_coeffs(wavelength_range, mode, return_sigmas=True)
        elif LD_type == 'Kipping':
                   (u1[i], u2[i]), (err_u1[i], err_u2[i]) =  sld.compute_kipping_ld_coeffs(wavelength_range, mode, return_sigmas=True)
        elif LD_type == 'uniform':	
                  hold = sld.compute_linear_ld_coeffs(wavelength_range, mode, return_sigmas=True)  
                  u1[i] = np.array(hold)[0]
                  err_u1[i] = np.array(hold)[1] 
    except:
        print('something')

if LD_type == 'quadratic' or LD_type == 'Kipping':
	df = pd.DataFrame({'wv':wavelength/10000, 'u1': u1,'err_u1':err_u1, 'u2': u2, 'err_u2': err_u2,})
	df.to_csv('/Users/kenearnold/Desktop/B_group/limb_darkening/limb_darkening_' + filters + '.txt', index=False, sep='\t')
	df.to_csv('limb_darkening_hst.txt', index=False, sep='\t')
elif LD_type == 'uniform':
	df = pd.DataFrame({'wv':wavelength/10000, 'u1': u1,'err_u1':err_u1})
	df.to_csv('/Users/kenearnold/Desktop/B_group/limb_darkening/limb_darkening_' + filters + '.txt', index=False, sep='\t')
	df.to_csv('limb_darkening_hst.txt', index=False, sep='\t')

