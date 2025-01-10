import pandas as pd
import numpy as np
import data_extraction
import pdb

def initialize(priors_to_apply, fit_bounds, planet, inst_to_fit, wavelength_range, broadband, data_extract, make_figures):
    
    priors = pd.read_csv('./priors/priors_' + planet + '.csv', sep=',', index_col=[0], header=[0])
    priors_to_apply = {}
    norm_length={}
    filenames={}
    errorfile={}
    
    for filters in inst_to_fit:
        
        fit_bounds[filters] = pd.read_csv('./priors''/fit_bounds_' + filters + '.csv', sep=',',  index_col=[0], header=[0])
        
        if broadband == True:
            priors_to_apply[filters]= pd.read_csv('./priors_to_apply/' + filters + '_BB.csv', sep=',', index_col=[0], header=[0])
        else:    
            priors_to_apply[filters]= pd.read_csv('./priors_to_apply/' + filters + '.csv', sep=',', index_col=[0], header=[0])

    if 'NIRCam444' in inst_to_fit:
        
        jwst_filter = 'NIRCam444'
        
        wavelength_range['NIRCam444'] = {}
        
        wv_min = 3.8975 #microns
        wv_max = 5.0525 #microns
        wavelength_range['NIRCam444'] = np.arange(wv_min, wv_max + (0.015/2), 0.015)

        if broadband == True:
            wavelength_range['NIRCam444'] = [np.mean(wavelength_range['NIRCam444'])]
        
        num_of_channels = int(len(wavelength_range['NIRCam444']))
            
        if data_extract == True:
            window_len = 20
            bg_y1=int(5)
            bg_y2=int(70)
            bg_y3=int(180)
            bg_y4=int(200)
            bkg1_start = int(660)
            bkg1_end = int(2048)
            bkg2_start = int(900)
            bkg2_end = int(2048)

            bg_thresh=[4, 4]
            median_thresh = 8
            p7thresh = 5 #sigma threshold
            star_center = 29
            
            data_extraction.data_extraction(jwst_filter, num_of_channels, wv_min, wv_max, make_figures, window_len, bg_y1, bg_y2, bg_y3, bg_y4,  bkg1_start, bkg1_end, bkg2_start, bkg2_end, bg_thresh, median_thresh, p7thresh, star_center)
                
        norm_length['NIRCam444'] = 240
    
        if broadband == True:
            filenames['NIRCam444'] = './DataAnalysis/jwst/' + planet + '/' + jwst_filter + '/flux_BB.txt'
            errorfile['NIRCam444'] = './DataAnalysis/jwst/' + planet + '/'  + jwst_filter + '/error_BB.txt'
        else:
            #filenames['NIRCam444'] = './DataAnalysis/jwst/wasp80b/eureka/' + jwst_filter + '/S4_wasp80b_ap9_bg14_LCData.h5'#
            filenames['NIRCam444'] = './DataAnalysis/jwst/' + planet + '/' + jwst_filter + '/flux.txt'
            errorfile['NIRCam444'] = './DataAnalysis/jwst/' + planet + '/' + jwst_filter + '/error.txt'
            
    if 'NIRCam322' in inst_to_fit:
        
        jwst_filter = 'NIRCam322'
        
        wavelength_range['NIRCam322'] = {}
        
        wv_min = 2.4575 #microns
        wv_max = 3.9425 
        wavelength_range['NIRCam322'] = np.arange(wv_min, wv_max + (0.015/2), 0.015)
            
        if broadband == True:
            wavelength_range['NIRCam322'] = [np.mean(wavelength_range['NIRCam322'])]
        
        num_of_channels = int(len(wavelength_range['NIRCam322']))
        
        if data_extract == True:
            window_len = 20
            bg_y1=int(12)
            bg_y2=int(52)
            bg_y3=int(200)
            bg_y4=int(220)
            bkg1_start = int(0)
            bkg1_end = int(1790)
            bkg2_start = int(0)
            bkg2_end = int(1790)

            bg_thresh=[4, 4]
            median_thresh = 8
            p7thresh = 5 #sigma threshold
            star_center = 32

            data_extraction.data_extraction(jwst_filter, num_of_channels, wv_min, wv_max, make_figures, window_len, bg_y1, bg_y2, bg_y3, bg_y4, bkg1_start, bkg1_end, bkg2_start, bkg2_end,bg_thresh, median_thresh, p7thresh, star_center)
             
        norm_length['NIRCam322'] = 445
        
        if broadband == True:
            filenames['NIRCam322'] = './DataAnalysis/jwst/' + planet + '/'  + jwst_filter + '/flux_BB.txt'
            errorfile['NIRCam322'] = './DataAnalysis/jwst/' + planet + '/' + jwst_filter + '/error_BB.txt'
        else:
            filenames['NIRCam322'] = './DataAnalysis/jwst/' + planet + '/' + jwst_filter + '/flux.txt'
            errorfile['NIRCam322'] = './DataAnalysis/jwst/' + planet + '/' + jwst_filter + '/error.txt'
        
    if 'MIRI' in inst_to_fit:
        
        filenames['MIRI']= './Data/' + planet + '/MIRI/S4_wasp80b_transit_ap3_bg9_LCData.h5'
        
        wavelength_range['MIRI'] = {}
        if broadband == False: 
            num_of_channels = 1
            wavelength_range['MIRI'] = np.zeros(num_of_channels)
        else: 
            num_of_channels = 1
        
    if 'WFC3' in inst_to_fit:
        
        #filenames['WFC3'] = './DataAnalysis/jwst/' + planet + '/'  + jwst_filter + '/ '
        
        #wavelength = np.linspace(1.115, 1.685, 20)
        wavelength = (['1.148', '1.173', '1.198', '1.224', '1.249', '1.274', '1.300', '1.325', '1.350', '1.376', '1.401', '1.427', '1.452', '1.477', '1.503', '1.528', '1.553', '1.579', '1.604', '1.629'])
        
        wavelength_range['WFC3'] = {}
        if broadband == False: 
            num_of_channels = 20
            wavelength_range['WFC3'] = wavelength
        else: 
            num_of_channels=1
            wavelength_range['WFC3'] = [1.3885]

            pass
                
    if 'shortwave444' in inst_to_fit:
        
        filenames['shortwave444'] = './DataAnalysis/jwst/wasp80b/shortwave/phot_auto_params_001_prog01185103001_WASP-80_F444W.fits'
        wavelength_range['shortwave444'] = {}
        
        wv_min = 1.963 #microns
        wv_max = 2.232	#microns
        wavelength_range['shortwave444'] = np.arange(wv_min, wv_max, 0.015)
        num_of_channels = 1
        if broadband == True:
            num_of_channels = 1
            wavelength_range['shortwave444'] = [np.mean(wavelength_range['shortwave444'])]
        
        norm_length['shortwave444'] = 240
    
    
    if 'shortwave322' in inst_to_fit:
        
        filenames['shortwave322'] = './DataAnalysis/jwst/wasp80b/shortwave/phot_p003_roebak_spatial_cleaning_prog01185_obs02_wasp80.fits'
        
        wavelength_range['shortwave322'] = {}
        
        wv_min = 1.963 #microns
        wv_max = 2.232	#microns
        wavelength_range['shortwave322'] = np.arange(wv_min, wv_max, 0.015)
                
        if broadband == True:
            num_of_channels = 1
            wavelength_range['shortwave322'] = [np.mean(wavelength_range['shortwave322'])]
        
        norm_length['shortwave322'] = 445

    
    return priors, priors_to_apply, fit_bounds, filenames, errorfile, wavelength_range, num_of_channels, norm_length