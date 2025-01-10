import xarray as xr
import numpy as np
from astropy.io import fits
import pandas as pd
import pdb

def pull_data_dat(filename, i, make_CF, inst_to_fit):
    
    df = pd.read_csv(filename, delim_whitespace=True, names=['time', 'Flux', 'err'])
    time = df['time'] - 2400000.5
    flux =  df['Flux']
    err = df['err']
    return time, flux, err

def pull_data_h5(filename, i, make_CF, inst_to_fit):

    lc_S4 = xr.load_dataset(filename,  engine='netcdf4')
    
    time = lc_S4.time 
    time = time.data
    
    flux_S4 = lc_S4.data[i,:]
    data = flux_S4.data
        
    err = lc_S4.err[i,:]
    sigma = err.data
    
    sigma = sigma[~(np.isnan(data))]
    time = time[~(np.isnan(data))]
    data = data[~(np.isnan(data))]
            
    norm = np.nanmedian(data)
    data = (data/norm)
        
    if make_CF == False:
        df = pd.read_csv('./correction_factor/correction_factor_' + inst_to_fit + '.txt', sep='\t')
        sigma =  (sigma/norm)*df['0'][0]
    else:
        sigma = (sigma/norm)
        
    wavelength = np.array(lc_S4.wavelength)

    return time, data, sigma, wavelength

def pull_data_txt(filename, errorfile, i, norm_length, make_CF, inst_to_fit):
    
    total_flux = pd.read_csv(filename, delim_whitespace=True)#, skiprows=1, names=['0', 'time'])
    total_error = pd.read_csv(errorfile, delim_whitespace=True)#, skiprows=1, names=['0', 'time'])
    
    data = total_flux[str(np.int64(i))]
    data = np.float64(data)
    sigma = total_error[str(np.int64(i))]
    sigma = np.float64(sigma)
    time = total_flux['time']
    time = np.float64(time)
        
    #pdb.set_trace()
    sigma = sigma[~(np.isnan(data))]
    time = time[~(np.isnan(data))]
    data = data[~(np.isnan(data))]
    
    norm = np.nanmedian(data[0:norm_length])
    data = (data/norm)
    
    if make_CF == False:
        df = pd.read_csv('./correction_factor/correction_factor_' + inst_to_fit + '.txt', sep='\t')
        sigma =  (sigma/norm)*df['0'][0]
    else:
        sigma = (sigma/norm)
    
    
    return time, data, sigma 

def pull_data_npz(filename_hst, filename_hst_bb, broadband, i, make_CF, inst_to_fit):
    

    data_frame = np.load(filename_hst)

    data = data_frame['y']
    error = data_frame['yerr']
    time = data_frame['times'] - 2400000.5
    
    
    if broadband == True:

        data_frame_bb = np.load(filename_hst_bb)

        data = data_frame_bb['y']
        error = data_frame_bb['yerr']
        time = data_frame_bb['times'] - 2400000.5
        
        data = data
        sigma = error
        wavelength = [np.mean(data_frame['wavelengths'])]
        
    else:
        data = data[i,:]
        sigma = error[i,:]
        wavelength = data_frame['wavelengths']
    
    if make_CF == False:
        df = pd.read_csv('./correction_factor/correction_factor_' + inst_to_fit + '.txt', sep='\t')
        sigma =  (sigma)*df['0'][0]
    
    return time, data, sigma, wavelength

def pull_data_short(filename, norm_length, make_CF, inst_to_fit):
    
    hdul = fits.open(filename)
    data = hdul[0].data
    sigma = hdul[1].data
    time = hdul[3].data - 2400000.5
    
    #pdb.set_trace()
    norm = (np.median(data[0:norm_length]))
        
    if make_CF == False:
        df = pd.read_csv('./correction_factor/correction_factor_' + inst_to_fit + '.txt', sep='\t')
        sigma =  (sigma/norm)*df['0'][0]
        pass
    else:
        sigma = (sigma/norm)
        
    data = data/norm
    sigma = sigma/norm
    
    #pdb.set_trace()
    
    return time, data, sigma

def pull_data_fits(filename):
    
    hdul = fits.open(filename)
    data = hdul[1].data
    
    time = data.TIME
    
    t_start = 59769.40305339

    t_stop = 59795.63381860

    time = np.linspace(t_start, t_stop, len(time))
    
    flux = data.PDCSAP_FLUX
    err = data.PDCSAP_FLUX_ERR
    
    i = np.where(np.isnan(flux))[0]
    time = np.delete(time, i)
    flux = np.delete(flux, i)
    err = np.delete(err, i)
    
    N = np.median(flux)
    
    data = flux/N
    sigma = err/N
    
    #data = data[7800:8015]
    #time = time[7800:8015]
    #sigma = sigma[7800:8015]
    
    data = data[2000:10000]
    time = time[2000:10000]
    sigma = sigma[2000:10000]

    return time, data, sigma

def pull_data_hst(filename, i, make_CF, inst_to_fit, visit_to_fit):
        
    total_flux = pd.read_csv(filename, skiprows=14, delim_whitespace=True)
        
    data = np.float64(total_flux['spec_opt'][0:len(total_flux['spec_opt'])])
    time = np.float64(total_flux['t_bjd'][0:len(total_flux['spec_opt'])])
    sigma = np.float64(np.sqrt((total_flux['var_opt'][0:len(total_flux['spec_opt'])])))
    scan = np.float64(total_flux['scan'][0:len(total_flux['scan'])])
    visit = np.float64(total_flux['ivisit'][0:len(total_flux['ivisit'])])
    orbit = np.float64(total_flux['iorbit'][0:len(total_flux['iorbit'])])
    
    remove_orbit = min(np.where(visit == visit_to_fit)[0])

    index1 = np.where(visit != visit_to_fit)[0]

    index2 = np.where((orbit == orbit[remove_orbit]))[0]

    data = np.delete(data, np.concatenate([index1, index2, ]))
    time = np.delete(time, np.concatenate([index1, index2, ]))
    sigma = np.delete(sigma, np.concatenate([index1, index2, ]))
    visit = np.delete(visit, np.concatenate([index1, index2, ]))
    scan =  np.delete(scan, np.concatenate([index1, index2, ]))
    
    if visit_to_fit == 2:
        data = np.delete(data, 51)
        time = np.delete(time, 51)
        sigma = np.delete(sigma, 51)
        visit = np.delete(visit, 51)
        scan = np.delete(scan, 51)
    if visit_to_fit == 5:
        data = np.delete(data, 30)
        time = np.delete(time, 30)
        sigma = np.delete(sigma, 30)
        visit = np.delete(visit, 30)
        scan = np.delete(scan, 30)

    index4 = np.where((scan == 0.0) & (visit == visit_to_fit))[0]
    index5 = np.where((scan == 1.0) & (visit == visit_to_fit))[0]
    
    sigma[index4] = sigma[index4]/np.nanmedian(data[index4])
    sigma[index5] = sigma[index5]/np.nanmedian(data[index5])
    
    data[index4] = data[index4]/np.nanmedian(data[index4])
    data[index5] = data[index5]/np.nanmedian(data[index5])
                
    sigma = sigma[~(np.isnan(data))]
    time = time[~(np.isnan(data))]
    data = data[~(np.isnan(data))]
    
    if make_CF == False:
        df = pd.read_csv('./correction_factor/correction_factor_' + inst_to_fit + '.txt', sep='\t')
        sigma =  (sigma)*df['0'][i]

    return time, data, sigma
    
    
    
    
    
