import numpy as np
from astropy.io import fits
import pdb
import matplotlib.pyplot as plt
from tqdm import tqdm 
import pandas as pd
import scipy.interpolate as spi
import scipy.ndimage as spn
from astropy.stats import sigma_clip
import os
from scipy.interpolate import griddata
from astropy.convolution import Box1DKernel, convolve
from photutils import Background2D, MedianBackground, BkgZoomInterpolator
from astropy.stats import SigmaClip
import warnings

warnings.filterwarnings('ignore', '.*Input data*', )


#import matplotlib
#from photutils.detection import DAOStarFinder
#from photutils.psf import PSFPhotometry, IntegratedGaussianPRF

os.environ["CRDS_PATH"] = '{}/crds_cache'.format(os.environ.get('HOME')) 
os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"

os.system('export CRDS_PATH=$HOME/crds_cache')
os.system('export CRDS_SERVER_URL=https://jwst-crds.stsci.edu')

def medstddev(data, mask=None, medi=False, axis=0):
    # mask invalid values:
    data = np.ma.masked_invalid(data)
    if mask is not None:
        data = np.ma.masked_where(~mask.astype(bool), data)
    # number of good values:
    ngood = np.sum(~np.ma.getmaskarray(data), axis=axis)

    # calculate median of good values:
    median = np.ma.median(data, axis=axis)
    #REMOVING OUTLIERS IN MEDIAN FRAME
    for i in range(-1, len(median[:,0])-1):
        i+=1
        test = median[i,:]
        index = np.where(abs(test - np.mean(test))/np.std(test) > 3)
        median[i,:][index] = np.nan
        
    median = np.ma.masked_invalid(median)
    # residuals is data - median, masked values don't count:
    residuals = data - median
    # calculate standar deviation:
    with np.errstate(divide='ignore', invalid='ignore'):
        std = np.ma.std(residuals, axis=axis, ddof=1)

    # Convert masked arrays to just arrays
    std = np.array(std)
    median = np.array(median)
    if std.shape == ():
        # If just a single value, make sure using a shaped array
        std = std.reshape(-1)
        median = median.reshape(-1)

    # critical case fixes:
    if np.any(ngood == 0):
        std[np.where(ngood == 0)] = np.nan
        median[np.where(ngood == 0)] = np.nan
    if np.any(ngood == 1):
        std[np.where(ngood == 1)] = 0.

    if len(std) == 1:
        std = std[0]
        median = median[0]

    # return statement:
    if medi:
        return (std, median)
    return std

import warnings
warnings.filterwarnings("ignore")   
    
def flight_poly_grismr_nc(pixels,obsFilter,detectorPixels=False):
    """
    Flight polynomials for NIRCam GRISMR grism time series 
    
    Parameters
    ----------
    obsFilter: str
        NIRCam Observation filter: F322W2 or F444W
    detectorPixels: bool
        Are the pixels in detector pixels from raw fitswriter output?
        This should be False for the MAST products in DMS format
    """
    if detectorPixels == True:
        x = 2048 - pixels - 1
    else:
        x = pixels
    if obsFilter == 'NIRCam322':
        x0 = 1571.
        coeff = np.array([ 3.92693691e+00,  9.81165339e-01,  1.66653554e-03, -2.87412352e-03])
        xprime = (x - x0)/1000.
    elif obsFilter == 'NIRCam444':
        ## need to update once we know where the new F444W position lands
        x0 = 945
        xprime = (x - x0)/1000.
        coeff = np.array([3.928041104137344 + 0.091033325, 0.979649332832983])
    else:
        raise Exception("Filter {} not available".format(obsFilter))
        
    poly = np.polynomial.Polynomial(coeff)
    return poly(xprime)

def get_wave_bounds(wv_min, wv_max, jwst_filter, N):
        
    bounds = [0,0]
        
    if jwst_filter == 'NIRCam444':
        
        pixel = np.linspace(0, N-1, N)
        
        #xprime = (pixel - 852.0756) / 1000
        
        #wavelength_range = 3.928041104137344 + 0.979649332832983 * xprime
        wavelength_range = flight_poly_grismr_nc(pixel,'NIRCam444',detectorPixels=False)
        
        index_min = abs(wavelength_range - (wv_min - (0.015/2))).argmin()
        index_max = abs(wavelength_range - (wv_max + (0.015/2))).argmin()
        
        bounds[0] = int(pixel[index_min])
        bounds[1] = int(pixel[index_max])
                
    elif jwst_filter == 'NIRCam322':
        
        pixel = np.linspace(0, N-1, N)
        
        #xprime = (pixel - 852.0756) / 1000
        
        #wavelength_range = 3.928041104137344 + 0.979649332832983 * xprime
        wavelength_range = flight_poly_grismr_nc(pixel,'NIRCam322',detectorPixels=False)

        index_min = abs(wavelength_range - (wv_min - (0.015/2))).argmin()
        index_max = abs(wavelength_range - (wv_max + (0.015/2))).argmin()
        
        bounds[0] = int(pixel[index_min])
        bounds[1] = int(pixel[index_max])
                
    return bounds, wavelength_range, pixel
    
def standard_spectrum(apdata, apmask, aperr):
    
    print('Computing standard spectrum ...')

    # Replace masked pixels with spectral neighbors
    apdata_cleaned = np.copy(apdata)
    aperr_cleaned = np.copy(aperr)
    
    for t, y, x in np.array(np.where(apmask == 0)).T:
        # Do not extend to negative indices (short and long wavelengths
        # do not have similar profiles)
        lower = x-2
        if lower < 0:
            lower = 0
        # Get mask for current neighbors
        mask_temp = np.append(apmask[t, y, lower:x],
                              apmask[t, y, x+1:x+3])

        # Gather current data neighbors and apply mask
        replacement_val = mask_temp*np.append(apdata_cleaned[t, y, lower:x],
                                              apdata_cleaned[t, y, x+1:x+3])
        # Figure out how many data neighbors are being used
        denom = np.sum(mask_temp)
        # Compute the mean of the unmasked data neighbors
        replacement_val = np.nansum(replacement_val)/denom
        # Replace masked value with the newly computed data value
        apdata_cleaned[t, y, x] = replacement_val

        # Gather current err neighbors and apply mask
        replacement_val = mask_temp*np.append(aperr_cleaned[t, y, lower:x],
                                              aperr_cleaned[t, y, x+1:x+3])
        # Compute the mean of the unmasked err neighbors
        replacement_val = np.nansum(replacement_val)/denom
        # Replace masked value with the newly computed err value
        aperr_cleaned[t, y, x] = replacement_val


    # Compute standard spectra
    apspec = np.nansum(apdata_cleaned, axis=1)
    aperr = np.nansum(aperr_cleaned**2, axis=1)
    
    return apspec, aperr
   
def profile(meddata):

    # profile = np.copy(meddata*mask)
    profile = np.copy(meddata)
    
    # Enforce positivity
    profile[np.where(profile < 0)] = 0
    # Normalize along spatial direction
    with np.errstate(divide='ignore', invalid='ignore'):
        profile /= np.sum(profile, axis=0)

    return profile

def get_median_frame(data, error_array, mask, window_len, median_thresh):
    
    flux_ma = np.ma.masked_where(mask == 0, data)
    medflux = np.ma.median(flux_ma, axis=0)
    ny, nx = medflux.shape
    
    # Interpolate over masked regions
    interp_med = np.zeros((ny, nx))
    xx = np.arange(nx)
    for j in range(ny):
        y1 = xx[~np.ma.getmaskarray(medflux[j])]
        goodrow = medflux[j][~np.ma.getmaskarray(medflux[j])]
        if len(goodrow) > 0:
            f = spi.interp1d(y1, goodrow, 'linear',
                             fill_value='extrapolate')
            # f = spi.UnivariateSpline(y1, goodmed, k=1, s=None)
            interp_med[j] = f(xx)

    # Apply smoothing filter along dispersion direction
    smoothflux = spn.median_filter(interp_med, size=(1, window_len))
    # Compute median error array
    err_ma = np.ma.masked_where(mask == 0, error_array)
    mederr = np.ma.median(err_ma, axis=0)
    # Compute residuals in units of std dev
    residuals = (medflux - smoothflux)/mederr

    # Flag outliers
    #outliers = sigma_clip(residuals, sigma=median_thresh, maxiters=5, cenfunc=np.ma.median, stdfunc=np.ma.std)

    outliers = np.zeros(np.shape(medflux))
    outliers[outliers == 0] = None
    outliers = np.ma.masked_invalid(outliers)
    for i in range(-1, len(outliers)-1):
        i+=1
        outliers[i] = sigma_clip(residuals[i], sigma=median_thresh, maxiters=5, cenfunc=np.ma.median, stdfunc=np.ma.std)
    # Interpolate over bad pixels
    
    clean_med = np.zeros((ny, nx))
    xx = np.arange(nx)
    for j in range(ny):
        y1 = xx[~np.ma.getmaskarray(outliers[j]) *
                ~np.ma.getmaskarray(medflux[j])]
        goodrow = medflux[j][~np.ma.getmaskarray(outliers[j]) *
                             ~np.ma.getmaskarray(medflux[j])]
        f = spi.interp1d(y1, goodrow, 'linear', fill_value='extrapolate')
        # f = spi.UnivariateSpline(y1, goodmed, k=1, s=None)
        clean_med[j] = f(xx)

    # Assign cleaned median frame to data object
    medflux = clean_med
    
    return medflux

def check_nans(data, mask, name=''):

    data = np.ma.masked_where(mask == 0, np.copy(data))
    masked = np.ma.masked_invalid(data).mask
    inan = np.where(masked)
    num_nans = np.sum(masked)
    num_pixels = np.size(data)
    mask[inan] = 0
    
    return mask

def convert_to_e(data, error, v0, hdu):
    
    if "EFFINTTM" in hdu[0].header:
        int_time = hdu[0].header['EFFINTTM']
    elif "EXPTIME" in hdu[0].header:
        int_time = hdu[0].header['EXPTIME']

    data *= int_time
    error *= int_time
    v0 *= int_time

    return data, error, v0

def dn2electrons(data, error, v0, hdu):#, bounds, star_center, window_len):
     
    xstart = hdu[0].header['SUBSTRT1']
    ystart = hdu[0].header['SUBSTRT2']
    nx = hdu[0].header['SUBSIZE1']#len(data[0][0,:])#
    ny = hdu[0].header['SUBSIZE2']#len(data[0][:,0])#
    
    # Load gain array in units of e-/ADU
    gain_header = fits.getheader('jwst_nircam_gain_0097.fits')
    xstart_gain = gain_header['SUBSTRT1']
    ystart_gain = gain_header['SUBSTRT2']

    ystart_trim = ystart-ystart_gain# + 1#int((ystart-ystart_gain + 4) + (star_center - (window_len/2)))# 1 indexed, NOT zero
    xstart_trim = xstart-xstart_gain# + 1#int(xstart-xstart_gain + 4 + bounds[0])
    
    gain = fits.getdata('jwst_nircam_gain_0097.fits')[ystart_trim:ystart_trim+ny,
                                       xstart_trim:xstart_trim+nx]    
    data *= gain
    error *= gain
    v0 *= (gain)**2
    
    data, error, v0 = convert_to_e(data, error, v0, hdu)
    
    return data, error, v0
    
def sigrej(data, sigma, mask, axis=0):
    
    # Get sizes
    dims = list(np.shape(data))
    nsig = np.size(sigma)
    if nsig == 0:
        nsig = 1
        sigma = [sigma]

    # Remove axis
    del (dims[axis])
    ival = np.empty((2, nsig) + tuple(dims))
    ival[:] = np.nan

    # Iterations
    for iter in np.arange(nsig):

        ival[1, iter], ival[0, iter] = medstddev(data, mask, axis=axis, medi=True)
        # if estsig[iter] > 0:   # if we dont have an estimated std dev.

        # Update mask
        # note: ival is slicing
        
        mask *= ((data >= (ival[0, iter] - sigma[iter] * ival[1, iter])) &
                 (data <= (ival[0, iter] + sigma[iter] * ival[1, iter])))

    # the return arrays
    ret = (mask,)

    if len(ret) == 1:
        return ret[0]
    return ret

def flag_bg(data, mask, bg_y1, bg_y2, bg_y3, bg_y4, bkg1_start, bkg1_end, bkg2_start, bkg2_end, bg_thresh):
        
    data_hold = np.copy(data)
    
    mask_hold = np.copy(mask)
    
    mask_hold[:, bg_y1:bg_y2, bkg1_start:bkg1_end] = False
    mask_hold[:, bg_y3:bg_y4, bkg2_start:bkg2_end] = False
    
    mask_hold = np.invert(mask_hold)
    
    data_hold = np.ma.masked_invalid(data_hold)
    data_hold.mask = mask_hold
    
   #bgdata1 = data_hold[:, :bg_y1]
    #bgmask1 = mask_hold[:, :bg_y1]
    #bgdata2 = data_hold[:, bg_y2:bg_y3]
    #bgmask2 = mask_hold[:, bg_y2:bg_y3]
    #bgdata3 = data_hold[:, bg_y4:]
    #bgmask3 = mask_hold[:, bg_y4:]
                
    mask_hold = np.invert(mask_hold)

    mask = sigrej(data_hold, bg_thresh, mask_hold)
    #mask[:, bg_y2:bg_y3] = sigrej(bgdata2, bg_thresh, bgmask2)
    #mask[:, bg_y4:] = sigrej(bgdata3, bg_thresh, bgmask3)
    
    mask[:, bg_y1:bg_y2, bkg1_start:] = True
    mask[:, bg_y3:bg_y4, bkg2_start:] = True 
    
    return data, mask

def get_pixel_from_wv(true_wv, pixels, jwst_filter):
    
    test_wv = flight_poly_grismr_nc(pixels,jwst_filter,detectorPixels=False)
    
    ind = abs(test_wv - true_wv).argmin()
    
    test_pixels = np.linspace(pixels[ind-1], pixels[ind+1], 50000)
    
    find_closest_pixel = flight_poly_grismr_nc(test_pixels,jwst_filter,detectorPixels=False)
    
    ind = abs(find_closest_pixel - true_wv).argmin()
    
    true_pixel = test_pixels[ind]
    
    return true_pixel - pixels[0]
    
def bin_spectra(optimal_spectrum, jwst_filter, variance, channels, bounds, true_wavelengths, make_figures):
    
    pixels = np.linspace(0, int(bounds[1] - bounds[0]), int(bounds[1] - bounds[0])+1) + int(bounds[0])
    true_bounds=[0,0]
    
    true_bounds[0] = get_pixel_from_wv(true_wavelengths[0], pixels, jwst_filter)
    true_bounds[1] = get_pixel_from_wv(true_wavelengths[int(channels-1)], pixels, jwst_filter)

    broadband_window = (true_bounds[1] - true_bounds[0])
    
    window = (broadband_window/channels)
    
    j = -window/2
                   
    higher_bleedx = 0.0
    
    binned_spectrum = np.zeros(channels)
    binned_error = np.zeros(channels)
    
    wv = np.zeros(channels)
    
    for i in range(-1, channels-1):
        
        j += (window)
        i+=1
        
        true_wv = true_wavelengths[i]
        
        true_pixel = get_pixel_from_wv(true_wv, pixels, jwst_filter)
        
        j = true_pixel
        
        upper_bound = j + (window/2)
        lower_bound = j - (window/2)
        
        if round(upper_bound) - upper_bound < 0:
            higher_bleedx = upper_bound - round(upper_bound)
            end = int(upper_bound)
        elif upper_bound - round(upper_bound) < 0:
            higher_bleedx = 1 - (round(upper_bound) - upper_bound)
            end = round(upper_bound)
            
        if round(lower_bound) - lower_bound < 0:
            lower_bleedx = 1 - (lower_bound - round(lower_bound))
            start = round(lower_bound)
        elif lower_bound - round(lower_bound)< 0:
            lower_bleedx = round(lower_bound) - lower_bound
            start = int(lower_bound)
            
        spectral_window = np.copy(optimal_spectrum[start:end])
        error_window = np.copy(variance[start:end])
   
        if len(spectral_window) > 16:
            pdb.set_trace()
                    
        spectral_window[0] = spectral_window[0]*lower_bleedx
        spectral_window[len(spectral_window)-1] = spectral_window[len(spectral_window)-1]*higher_bleedx
        
        error_window[0] = error_window[0]*lower_bleedx
        error_window[len(spectral_window)-1]*higher_bleedx
        
        binned_spectrum[i] = np.nansum(spectral_window)
        binned_error[i] = np.sqrt(np.nansum(error_window**2))
        
        if make_figures['binning_spectra'] == True and j == 0:
            plt.figure()
            plt.plot(pixels, optimal_spectrum)
            
            plt.vlines(x = j-(window/2), ymin = -1000000, ymax = 1000000,
                       colors = 'red', linewidth=1.0)
            
            plt.vlines(x = j+(window/2), ymin = -1000000, ymax = 1000000,
                       colors = 'red', linewidth=1.0)
            
            plt.ylim([min(optimal_spectrum) - 1000, max(optimal_spectrum)+1000])
            
            plt.title('Channel = ' + str(i))
        
        wv[i] = flight_poly_grismr_nc(j + pixels[0],jwst_filter,detectorPixels=False)
                    
    return binned_spectrum, binned_error, wv

def Column_BKG_fit(dataim, mask, x1, x2, x3, x4, deg=1, threshold=7):

    # rotating so that x is the spatial direction and y is the wavelength direction
    dataim = dataim.T
    mask = mask.T
        
    #masking the signal and the background star
    ny, nx = np.shape(dataim)
    if isinstance(x1, (int, np.int64)):
        x1 = np.zeros(ny, dtype=int)+x1
    if isinstance(x2, (int, np.int64)):
        x2 = np.zeros(ny, dtype=int)+x2
    if isinstance(x3, (int, np.int64)):
        x3 = np.zeros(ny, dtype=int)+x3
    if isinstance(x4, (int, np.int64)):
        x4 = np.zeros(ny, dtype=int)+x4
        
    degs = np.ones(ny)*deg
    # Initiate background image with zeros
    bg = np.zeros((ny, nx))
    # Fit polynomial to each column
    for j in range(ny):
        nobadpixels = False
        # Create x indices for background sections of frame
        xvals = np.concatenate((range(x1[j]),
                                range(x2[j], x3[j]), 
                                range(x4[j]+1, nx))).astype(int)
        # If too few good pixels then average
        too_few_pix = (np.sum(mask[j, :x1[j]]) < deg
                       or np.sum(mask[j, x2[j]:x3[j]]) < deg
                       or np.sum(mask[j, x4[j]+1:nx]) < deg)
        if too_few_pix:
            degs[j] = 0
        while not nobadpixels:
                
            goodxvals = xvals[np.where(mask[j, xvals])]

            dataslice = dataim[j, goodxvals]
            # Check for at least 1 good x value
            if len(goodxvals) == 0:
                nobadpixels = True 
            else:
                # Fit along spatial direction with a polynomial 
                
                coeffs = np.polyfit(goodxvals, dataslice, deg=degs[j])

                model = np.polyval(coeffs, goodxvals)

                residuals = dataslice - model

                stdres = np.std(residuals)

                if stdres == 0:
                    stdres = np.inf
                stdevs = np.abs(residuals) / stdres
                # Find worst data point
                loc = np.argmax(stdevs)
                # Mask data point if > threshold
                if stdevs[loc] > threshold:
                    mask[j, goodxvals[loc]] = 0
                else:
                    nobadpixels = True  # exit while loop

        # Evaluate background model at all points, write model to
        # background image
        if len(goodxvals) != 0:
            bg[j] = np.polyval(coeffs, range(nx))

    bg = (bg.T)
    mask = (mask.T)

    return bg, mask

def interp_masked(data, mask):
    
    flux = data
    nx = flux.shape[1]
    ny = flux.shape[0]
    grid_x, grid_y = np.mgrid[0:ny-1:complex(0, ny), 0:nx-1:complex(0, nx)]
    points = np.where(mask == 1)
    # x,y positions of not masked pixels
    points_t = np.array(points).transpose()
    values = flux[np.where(mask == 1)]  # flux values of not masked pixels

    # Use scipy.interpolate.griddata to interpolate
    grid_z = griddata(points_t, values, (grid_x, grid_y), method='linear')

    data = grid_z

    return data, mask

def replace_moving_mean(data, outliers, kernel, boundary):
    # First set outliers to NaN so they don't bias moving mean
   data[outliers] = np.nan
   smoothed_data = convolve(data, kernel, boundary)
   # Replace outliers with value of moving mean
   data[outliers] = smoothed_data[outliers]

   return data

def clip_outliers(data, mask, sigma, box_width, maxiters, boundary, fill_value):
    
    data = np.ma.masked_invalid(np.ma.copy(data))
    data = np.ma.masked_where(mask, data)

    kernel = Box1DKernel(box_width)

    outliers = np.zeros_like(data, dtype=bool)
    new_clipped = True
    i = 0
    while i < maxiters and new_clipped:
        i += 1

        # Compute the moving mean
        bound_val = np.ma.median(data)  # Only used if boundary=='fill'
        smoothed_data = convolve(data, kernel, boundary=boundary, fill_value=bound_val)
        # Compare data to the moving mean (to remove astrophysical signals)
        residuals = data-smoothed_data
        # Sigma clip residuals to find bad points in data
        residuals = sigma_clip(residuals, sigma=sigma, maxiters=maxiters,
                               cenfunc=np.ma.median)
        new_outliers = np.ma.getmaskarray(residuals)
        
        if np.all(new_outliers == outliers):
            new_clipped = False
        else:
            outliers = new_outliers
            data = np.ma.masked_where(outliers, data)
        
        if fill_value == 'mask':
            data = np.ma.masked_where(outliers, data)
        elif fill_value == 'mean':
            data = replace_moving_mean(data, outliers, kernel, boundary)
            outliers[:] = False
        
        return data, outliers, np.sum(outliers)

def plot_data(ax, lam, td, error, hold, x1, x2, y1, y2, n, bounds, label, color='green', name='Data', marker='o'):
    rolling=[]
    for i in range(n,len(lam)-n):
        rolling.append(np.mean(td[i-n:i+n+1]))

    ax[0].plot(lam[n:-n],np.asarray(rolling),color=color, alpha=1, lw=2, label=label)
    data_bins=np.abs(lam[1:]-lam[:-1])/2.0
    data_bins=np.append(data_bins,data_bins[-1])
    ax[0].set_ylim([min(np.asarray(rolling)) - 50, max(np.asarray(rolling))+50])
    ax[0].vlines(x=1536-bounds[0], ymin=-30000, ymax=30000, color='red')
    
    hold = np.nan_to_num(hold, nan=0.0)
    
    ax[1].imshow(hold, vmin=0, vmax=100)
    
    ax[0].tick_params(axis='both', which='major', labelsize=20)

    ax[1].tick_params(axis='both', which='major', labelsize=20)
    ax[1].invert_yaxis()
    
    ax[1].vlines(x = x1, ymin=y1, ymax=y2, color='red')
    ax[1].vlines(x = x2, ymin=y1, ymax=y2, color='red')
    ax[1].hlines(y = y1, xmin=x1, xmax=x2, color='red')
    ax[1].hlines(y = y2, xmin=x1, xmax=x2, color='red')
    ax[1].plot(1536 - bounds[0], y1 + ((y2-y1)/2), marker='x', color='red', markersize=30)
    ax[0].legend(loc='upper left', fontsize=10)
    ax[0].set_ylim(-10,10)
    
    ax[0].set_title('Rolling Average of BKG', fontsize=20)

def DataReduction(data_unmasked, flux_array, mask, bkg_array, k, l, jwst_filter):
    
    fontsize=20
    fig, ax = plt.subplots(3,1, figsize=[20, 10], sharex=True)
    fig.suptitle('Integration = ' + str(k+1) + '\n', fontsize=fontsize+10)
    
    plot_unmasked =  (np.copy(data_unmasked[k]))
    plot_unmasked = np.nan_to_num(plot_unmasked, nan=0.0)
    im1 = ax[0].imshow(plot_unmasked, vmin=0.0, vmax=300, aspect='3.5')
    ax[0].invert_yaxis()
    ax[0].set_ylim(0, 100)
    ax[0].tick_params(axis='both', which='major', labelsize=fontsize)
    
    plot_bkg = np.copy(bkg_array)
    plot_bkg = np.nan_to_num(plot_bkg)
    im2 = ax[2].imshow(plot_bkg, vmin=0.0, vmax=50, aspect='3.5')
    ax[2].invert_yaxis()
    ax[2].set_ylim(0, 100)
    ax[2].tick_params(axis='both', which='major', labelsize=fontsize)

    plot_mask = (np.copy(flux_array) - plot_bkg)
    plot_mask[np.isnan(plot_mask)] = 0.0
    mask_copy = np.copy(mask[l])
    plot_mask, mask_copy = (interp_masked(plot_mask, mask_copy))

    im3 = ax[1].imshow(plot_mask, vmin=0.0, vmax=300, aspect='3.5')
    ax[1].invert_yaxis()
    ax[1].set_ylim(0, 100)
    ax[1].tick_params(axis='both', which='major', labelsize=fontsize)
    
    ax[2].set_xlabel('Spectral-pixel', fontsize=fontsize+5)
    ax[1].set_ylabel('Spatial-pixel', fontsize=fontsize+5)
         
    cbar1 = fig.colorbar(im1, pad=0.01, aspect=5)
    cbar2 = fig.colorbar(im2, pad=0.01,  aspect=5)
    cbar3 = fig.colorbar(im3, pad=0.01, aspect=5)
    cbar1.set_label('electrons', fontsize=fontsize-5)
    cbar2.set_label('electrons', fontsize=fontsize-5)
    cbar3.set_label('electrons', fontsize=fontsize-5)

    plt.tight_layout(pad=0.1)

    plt.savefig('./figs/' + jwst_filter + '/DataReduction/bkg_subtraction.png', dpi=300, bbox_inches='tight')
      
def Spline_BKG_fit(image, make_figures, x1, x2, x3, x4, bkg1_start, bkg1_end, bkg2_start, bkg2_end,  spline_box=20, spline_order=3):
	
    mask = np.zeros(image.shape, dtype=bool)
    mask[x1:x2, bkg1_start:bkg2_end] = True
    mask[x3:x4, bkg2_start:bkg2_end] = True
        
    image = np.ma.MaskedArray(image, mask=mask)
    
    subimage = image[:, 0:511]  # amp 1
    submask = mask[:, 0:511]
    
    bkgd1 = Background2D(subimage, box_size=spline_box,
                      sigma_clip=SigmaClip(sigma=3.),
                      filter_size=3,
                      exclude_percentile=50,
                      bkg_estimator=MedianBackground(),
                      interpolator=BkgZoomInterpolator(order=spline_order),
                      mask=submask)
    
    subimage = image[:, 511:1023]  # amp 2
    submask = mask[:, 511:1023]
    
    bkgd2 = Background2D(subimage, box_size=spline_box,
                      sigma_clip=SigmaClip(sigma=3.),
                      filter_size=3,
                      exclude_percentile=50,
                      bkg_estimator=MedianBackground(),
                      interpolator=BkgZoomInterpolator(order=spline_order),
                      mask=submask)
    
    subimage = image[:, 1023:1535]  # amp 3    
    submask = mask[:, 1023:1535]
    
    bkgd3 = Background2D(subimage, box_size=spline_box,
                      sigma_clip=SigmaClip(sigma=3.),
                      filter_size=3,
                      exclude_percentile=50,
                      bkg_estimator=MedianBackground(),
                      interpolator=BkgZoomInterpolator(order=spline_order),
                      mask=submask)
    
    subimage = image[:, 1535:2048]  # amp 4
    submask = mask[:, 1535:2048]
    
    bkgd4 = Background2D(subimage, box_size=spline_box,
                      sigma_clip=SigmaClip(sigma=3.),
                      filter_size=3,
                      exclude_percentile=50,
                      bkg_estimator=MedianBackground(),
                      interpolator=BkgZoomInterpolator(order=spline_order),
                      mask=submask)
    
    bkgdlevel = np.hstack((bkgd1.background, bkgd2.background, bkgd3.background, bkgd4.background))

    if make_figures['bkg_subtraction'] == True:
        
        
        plt.imshow(bkgdlevel, origin='lower', vmin=0, vmax=50)
        plt.title("Model")
        
        plt.show()
        
    return bkgdlevel


def data_extraction(jwst_filter, channels, wv_min, wv_max, make_figures, window_len, bg_y1, bg_y2, bg_y3, bg_y4, bkg1_start, bkg1_end, bkg2_start, bkg2_end,  bg_thresh, median_thresh, p7thresh, star_center):
    
    true_wavelengths = np.arange(wv_min, wv_max + (0.015/2), 0.015)
    
    if jwst_filter == 'NIRCam322':
        filename = './Data/wasp80b/' + jwst_filter + '/jw01185002001_04103_00001-seg001_nrcalong_rateints.fits'
        
    elif jwst_filter == 'NIRCam444':
        filename = './Data/wasp80b/' + jwst_filter + '/jw01185103001_03104_00001-seg001_nrcalong_rateints.fits'
    
    hdu = fits.open(filename)
    data = hdu[1].data
    error = hdu[2].data
    
    nsegment = hdu[0].header['EXSEGTOT']
    
    time = np.zeros(hdu[0].header['NINTS'])
    
    k=-1
    
    N = len(data[0][0,:])
    
    bounds, wavelength_range, pixel = get_wave_bounds(wv_min, wv_max, jwst_filter, N)   
            
    apspec = np.zeros([hdu[0].header['NINTS'], round(window_len), bounds[1] - bounds[0]])
    aperror = np.zeros([hdu[0].header['NINTS'], round(window_len), bounds[1] - bounds[0]])
    background = np.zeros([hdu[0].header['NINTS'], round(window_len), bounds[1] - bounds[0]])
    apv0 = np.zeros([hdu[0].header['NINTS'], round(window_len), bounds[1] - bounds[0]])
    apmask = np.zeros([hdu[0].header['NINTS'], round(window_len), bounds[1] - bounds[0]])
    
    print('Stacking all frames and subtracting background ...')
    
    for m in range(0, nsegment):
        m += 1

        if jwst_filter == 'NIRCam322':
            filename = './Data/wasp80b/' + jwst_filter + '/jw01185002001_04103_00001-seg00' + str(m) + '_nrcalong_rateints.fits'
        elif jwst_filter == 'NIRCam444':
            filename = './Data/wasp80b/' + jwst_filter + '/jw01185103001_03104_00001-seg00' + str(m) + '_nrcalong_rateints.fits'
            #filename = './Data/wasp80b/' + jwst_filter + '/jw01185103001_03104_00001-seg00' + str(m) + '_nrcalong_BkgdSub_SplineRBR.fits'

        hdu = fits.open(filename)
        data = np.copy(hdu[1].data)
        error = np.copy(hdu[2].data)
        v0 = np.copy(hdu[6].data)
        
        mask = (np.ones(np.shape(data), dtype=bool))
        mask = check_nans(data, mask)
        
        data, mask = flag_bg(data, mask, bg_y1, bg_y2, bg_y3, bg_y4, bkg1_start, bkg1_end, bkg2_start, bkg2_end,  bg_thresh)
        
        data[mask == False] = np.nan
        error[mask == False] = np.nan
        v0[mask == False] = np.nan
        
        data, error, v0 = dn2electrons(data, error, v0, hdu)
        data_unmasked = np.copy(data)
        
        data = data[0:int(hdu[0].header['INTEND'] - hdu[0].header['INTSTART'])+1, 4:len(data[0][0,:])]
        data_unmasked = data_unmasked[0:int(hdu[0].header['INTEND'] - hdu[0].header['INTSTART'])+1, 4:len(data[0][0,:])]
        error = error[0:int(hdu[0].header['INTEND'] - hdu[0].header['INTSTART'])+1, 4:len(data[0][0,:])]
        v0 = v0[0:int(hdu[0].header['INTEND'] - hdu[0].header['INTSTART'])+1, 4:len(data[0][0,:])]
        mask = mask[0:int(hdu[0].header['INTEND'] - hdu[0].header['INTSTART'])+1, 4:len(data[0][0,:])]
               
        data = np.ma.masked_invalid(data)
        error = np.ma.masked_invalid(error)
        v0 = np.ma.masked_invalid(v0)
        
        for l in tqdm(range(-1, (hdu[0].header['INTEND'] - hdu[0].header['INTSTART']))):
            
            l += 1
            k+=1
            time[k] = hdu[4].data[l][5]
                        
            flux_array = np.copy(data[l])
            error_array = np.copy(error[l])
            v0_array = np.copy(v0[l])

            # = fitbg(flux_array, mask[l], bg_y1, bg_y2, bg_y3, bg_y4, deg=1, threshold=7)                        
            
            bkg_array = Spline_BKG_fit(flux_array, make_figures, bg_y1, bg_y2, bg_y3, bg_y4, bkg1_start, bkg1_end, bkg2_start, bkg2_end, spline_box=20, spline_order=3)                  
            
            apspec[k] = flux_array[int(star_center-round(window_len/2)):int(star_center+round(window_len/2)), int(bounds[0]):int(bounds[1])]
            aperror[k] = error_array[int(star_center-round(window_len/2)):int(star_center+round(window_len/2)),  int(bounds[0]):int(bounds[1])]
            background[k] = bkg_array[int(star_center-round(window_len/2)):int(star_center+round(window_len/2)),  int(bounds[0]):int(bounds[1])]
            apv0[k] = v0_array[int(star_center-round(window_len/2)):int(star_center+round(window_len/2)), int(bounds[0]):int(bounds[1])]
            apmask[k] = mask[l][int(star_center-round(window_len/2)):int(star_center+round(window_len/2)),  int(bounds[0]):int(bounds[1])]
                        
            if make_figures['DataReduction'] == True and k == 3:
                DataReduction(data_unmasked, flux_array, mask, bkg_array, k, l, jwst_filter)
    
    apmask = np.array(apmask, dtype=bool)
    
    apspec = apspec - background
    
    #apspec, aperror, apv0 = dn2electrons(apspec, aperror, apv0, hdu, bounds,  star_center, window_len)
        
    #STEP 4: Extract a standard spectrum 
    spectrum, error = standard_spectrum(apspec, apmask, aperror)
    
    apmedian = get_median_frame(apspec, aperror, apmask, int(11), median_thresh)

    P = profile(apmedian)
    
    #STEP 5: Construct spatial profile
    
    Q = 1 # Gain is one because we converted to electrons
    
    optimal_spectrum = np.zeros([hdu[0].header['NINTS'], len(apspec[0][0,:])])
    specvar = np.zeros([hdu[0].header['NINTS'], len(apspec[0][0,:])])
        
    print('STEP 6 - 8 in optimal extraction ...')
    # Loop through steps 6-8 until no more bad pixels are uncovered
    
    ii = 0
    bad_points = 1
    
    while bad_points > 0:
        ii += 1
        bad_points = 0

        print('iteration ' + str(ii)) 
        
        for i in tqdm(range(-1, hdu[0].header['NINTS']-1)):
            # STEP 6: Revise variance estimates
            i += 1
            
            expected = P*spectrum[i]
            variance = (np.abs(expected + background[i]) / Q) + apv0[i]
            # STEP 7: Mask cosmic ray hits
            stdevs = np.abs(apspec[i] - expected)*apmask[i]/np.sqrt(variance)
            apmask[i][np.isnan(stdevs)] = 0
                        
            for j in range(-1, len(apspec[0][0,:])-1):
                j+=1
                # Only continue if there are unmasked values
                if np.sum(apmask[i][:, j]) > 0:
                    # Find worst data point in each column
                    loc = np.nanargmax(stdevs[:, j])
                    # Mask data point if std is > p7thresh
                    if stdevs[loc, j] > p7thresh:     
                        bad_points += 1
                        apmask[i][loc, j] = False
                        
                        if sum(apmask[i][:, j]) < len(apspec[0][:,0])/2.:
                            apmask[i][:, j] = False

            # STEP 8: Extract optimal spectrum
            with warnings.catch_warnings():
                # Ignore warnings about columns that are completely masked
                warnings.filterwarnings(
                    "ignore", "invalid value encountered in")
                denom = np.nansum(P*P*apmask[i]/variance, axis=0)
                
            denom[np.where(denom == 0)] = np.inf
            
            spectrum[i] = np.nansum(P*apmask[i]*apspec[i]/variance, axis=0)/denom
            specvar[i] = np.sqrt(np.sum(P*apmask[i], axis=0) / denom)
            
            spectrum[i][spectrum[i] == 0.0] = np.nan
                
        spectrum = np.ma.masked_invalid(spectrum)
        specvar = np.ma.masked_invalid(specvar)
        apmask[np.where(spectrum.mask) == True] = False

        print('# of bad points ' + str(bad_points))
            
        optimal_spectrum = np.copy(spectrum)
        
    if make_figures['optimal_extraction'] == True:
        for i in range(-1, len(spectrum) - 1):
            i+=1
            plt.figure()
            plt.plot(spectrum[i])
            plt.title('integration = ' + str(i) + ' iteration = ' + str(ii))
    
    hold_spectrum = np.zeros([len(optimal_spectrum), channels])
    total_error = np.zeros([len(optimal_spectrum), channels])
    
    
    print('Binning spectral channels ...')
    for i in tqdm(range(-1, len(optimal_spectrum) - 1)):
        i += 1
        binned_spectrum, binned_error, wv = bin_spectra(optimal_spectrum[i], jwst_filter, specvar[i], channels, bounds, true_wavelengths, make_figures)
    
        hold_spectrum[i] = binned_spectrum
        total_error[i] = binned_error
        
    print('Removing outliers along time axis ...')
    sigma=4
    box_width = 50
    maxiters = 5
    boundary = 'fill'
    fill_value = 'mask'
    
    total_spectrum = np.zeros([len(optimal_spectrum), channels])

    for i in range(-1, len(total_spectrum[0,:])-1):
        i+=1
        
        mask = np.ma.masked_invalid(hold_spectrum[:,i]).mask
        
        test, outliers, num_of_outliers = np.copy(clip_outliers(hold_spectrum[:,i], mask, sigma, box_width, maxiters, boundary, fill_value))
        
        test[test.mask == True] = np.nan
        
        total_spectrum[:,i] = np.copy(test)
    
    total_spectrum=pd.DataFrame(data=total_spectrum)
    total_error=pd.DataFrame(data=total_error)
    
    total_spectrum.insert(channels, 'time', time, True)
        
    total_spectrum['time'] = time
    
    total_spectrum.to_csv('./DataAnalysis/jwst/wasp80b/' + jwst_filter + '/flux.txt', sep='\t', index=False, na_rep='NULL')
    total_error.to_csv('./DataAnalysis/jwst/wasp80b/' + jwst_filter + '/error.txt', sep='\t', index=False, na_rep='NULL')
            
    if make_figures['reduced_light_curve'] == True:
        plt.clf()
        for i in range(-1, channels-1):
            i+=1
            plt.figure()
            
            plot_flux = (total_spectrum[np.float64(i)]/np.nanmedian(total_spectrum[np.float64(i)]))
            plot_error = (total_error[np.float64(i)]/np.nanmedian(total_spectrum[np.float64(i)]))
            plot_error =  plot_error[~(np.isnan(plot_flux))]
    
            time1 = time[~(np.isnan(np.array(plot_flux)))]
            
            plot_flux =  plot_flux[~(np.isnan(plot_flux))]
            
            #pdb.set_trace()
            
            plt.errorbar(time1, plot_flux, yerr = [plot_error, plot_error], alpha=0.5, marker='.', linestyle='')
            #plt.scatter(time, plot_flux,  alpha=0.5)
            plt.xlabel('time (days)')
            plt.ylabel('Transit Depth (ppm)')
            #plt.ylim([0.96, 1.006])
            plt.xlim([min(time), max(time)])
            plt.title('bandpass = ' + str(i+1))
    else:
        pass
    
    plt.show()


              

            