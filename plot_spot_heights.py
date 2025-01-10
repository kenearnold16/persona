import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

wv_min = 3.8975 #microns
wv_max = 5.0525 #microns
wavelength = np.arange(wv_min, wv_max + (0.015/2), 0.015)

num_of_channels = int(len(wavelength))

spot_height = np.zeros(num_of_channels)
spot_err1 = np.zeros(num_of_channels)
spot_err2 = np.zeros(num_of_channels)

for i in range(-1, num_of_channels-1):
    i += 1
    filename_fits = './fit_variables/NIRCam444/fit_variables_table_' + str(i) + '.txt'
    filename_err1 = './fit_variables/NIRCam444/fit_error1_table_' + str(i) + '.txt'
    filename_err2 = './fit_variables/NIRCam444/fit_error2_table_' + str(i) + '.txt'
    
    df_fit = pd.read_csv(filename_fits, delim_whitespace=True)
    df_err1 = pd.read_csv(filename_err1, delim_whitespace=True)
    df_err2 = pd.read_csv(filename_err2, delim_whitespace=True)

    
    spot_height[i] = df_fit['NIRCam444']['spot_b']
    spot_err1[i] = df_err1['NIRCam444']['spot_b']
    spot_err2[i] = df_err2['NIRCam444']['spot_b']
    

fig, ax = plt.subplots(1,1, figsize=[20,8])
fontsize=20

ax.errorbar(wavelength, spot_height*1e6, yerr=[spot_err1*1e6, spot_err2*1e6], color='purple', linestyle='None', fmt='^')

ax.set_xlabel('wavelength ($\mu$m)', fontsize=fontsize)
ax.set_ylabel('spot height (ppm)',  fontsize=fontsize)
ax.set_title('WASP-80b F444W gaussian spot heights',  fontsize=fontsize)

ax.tick_params(axis='x', labelsize=fontsize)
ax.tick_params(axis='y', labelsize=fontsize)

ax.set_ylim(0, 0.002*1e6)

spots = np.vstack([spot_height, spot_err1, spot_err2]).transpose()

spot_heights = pd.DataFrame(spots, columns=['spot_heights', 'spot_err1', 'spot_err2'])


spot_heights.to_csv('./spot_heights/spot_heights_gaussian.txt')

plt.savefig('./spot_heights/spot_heights_gaussian.png', dpi=300)




    