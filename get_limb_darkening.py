import pandas as pd
import pdb
import numpy as np

def get_limb_darkening(broadband, filters, wavelength_range, i):
    
    if broadband == True:
        if filters == 'WFC3':
            LDC = pd.read_csv('./limb_darkening/limb_darkening_' + filters + '_BB.txt', delim_whitespace=True, skiprows=1, names=['wave_eff', 'u1', 'e1', 'u2', 'e2'])
        else:
            LDC = pd.read_csv('./limb_darkening/limb_darkening_' + filters + '_BB.txt', delim_whitespace=True, skiprows=2, names=['Teff', 'logg','FeH' , 'profile', 'filter', 'wave_min', 'wave_eff', 'wave_max', 'u1', 'e1' ,'u2', 'e2'])
        
        LDC_index = abs(np.float64(wavelength_range[filters])[i] - LDC['wave_eff']).argmin()
        u1 = LDC['u1'][LDC_index]
        u2 = LDC['u2'][LDC_index]    
        
        err_u1 = LDC['e1'][LDC_index]
        err_u2 = LDC['e2'][LDC_index]
        
    elif filters == 'WFC3':
        LDC = pd.read_csv('./limb_darkening/limb_darkening_' + filters + '.txt', delim_whitespace=True, skiprows=1, names=['wave_eff', 'u1', 'e1', 'u2', 'e2'])
        LDC_index = abs(np.float64(wavelength_range[filters])[i] - LDC['wave_eff']).argmin()
        u1 = LDC['u1'][LDC_index]
        u2 = LDC['u2'][LDC_index]    
        
        err_u1 = LDC['e1'][LDC_index]
        err_u2 = LDC['e2'][LDC_index]
        
    else:    
        LDC = pd.read_csv('./limb_darkening/' + filters + '_PolyShiftedOnlyq1_MPS2_Kipping2013.txt', delim_whitespace=True, names=['q1','q2'])
        #LDC = pd.read_csv('./limb_darkening/limb_darkening_' + filters + '.txt', delim_whitespace=True, skiprows=1, names=['wave_eff', 'u1', 'e1', 'u2', 'e2'])
        
        q1 = np.float64(LDC['q1'][i])
        q2 = np.float64(LDC['q2'][i])
        u1 = (q1**0.5)/(1+(4*q2))
        u2 = (q1**0.5)*(1 - (1/(1+(4*q2))))
        
        err_u1 = 0.01
        err_u2 = 0.01
    
    return u1, u2, err_u1, err_u2

