import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import initialize
import exoplanet as xo
import get_data as gd
import get_limb_darkening as gld
import theano.tensor as tt
import pdb


planet = 'wasp80b'

broadband=True
divide_white = True
data_extract=False
make_CF = True
star_spot_triangle = False
star_spot_ellipse = False
star_spot_gaussian = True

visit_to_fit = 7

make_figures = {}
make_figures['trace_extraction'] = False
make_figures['residuals'] = False
make_figures['find_center'] = False
make_figures['trace_fit'] = False
make_figures['transit_bounds'] = False
make_figures['reduced_light_curve'] = False
make_figures['check spectra'] = False
make_figures['check center'] = False
make_figures['bkg_subtraction'] = False
make_figures['shift_trace'] = False
make_figures['optimal_extraction'] = False
make_figures['binning_spectra'] = False
make_figures['save_LC_plots'] = True
make_figures['show_fit_model'] = False
make_figures['DataReduction'] = False
make_figures['corner_plots'] = False
make_figures['Rolling_BKG'] = False

inst_to_fit = ['shortwave444'] #['joint', 'NIRCam322','NIRCam444', 'MIRI', 'hst', 'shortwave322', 'shortwave444']
filters = inst_to_fit[0]

wavelength_range = {}
filenames = {}
norm_length={}
errorfile={}
priors_to_apply={}
fit_bounds={}
ramp_type={}

ramp_type['NIRCam444'] = 'line'
ramp_type['NIRCam322'] = 'line'
ramp_type['MIRI'] = 'exponential'
ramp_type['shortwave444'] = 'polynomial'
ramp_type['shortwave322'] = 'polynomial'

def ramp_scale(priors, filters, time, ramp_type):
    
     if ramp_type == 'line':
         median_time = np.median(time[filters])
         ramp = 1 + priors[filters]['ramp1'] + priors[filters]['slope']*(time[filters] - median_time)
     
     elif ramp_type == 'exponential':
         median_time = np.median(time[filters])
         t = (time[filters] - median_time)
         ramp = 1 + priors[filters]['ramp1']*np.exp(-(t)/priors[filters]['tau']) +  priors[filters]['slope']*t
         
     elif ramp_type == 'polynomial':
         median_time = np.median(time[filters])
         ramp = 1 + priors[filters]['ramp1'] + priors[filters]['ramp2']*(time[filters] - median_time) + priors[filters]['slope']*(time[filters] - median_time)**2
     
     return ramp

def spot_model_triangle(a, b, time, triangle_center):
    # Create a flat line initialized to zeros (Theano tensor)
    y = tt.zeros((len(time),))
        
    start_time = triangle_center - a
    end_time = triangle_center + a
    
    # Calculate the horizontal distance from the center
    time_distance = tt.abs_(time - triangle_center)
    
    # Compute the triangle shape using Theano operations
    triangle_value = b * (1 - (time_distance / a))
    
    # Only apply the triangle where x is within [start_time, end_time]
    spot = tt.switch((time >= start_time) & (time <= end_time), triangle_value, y)
    
    return spot

def spot_model_ellipse(a, b, time, ellipse_center):
    
    y = tt.zeros((len(time),))

    # Compute the boundaries of the semi-ellipse in time units
    start_time = ellipse_center - a
    end_time = ellipse_center + a
    
    # Calculate the horizontal position relative to the ellipse center
    x_ellipse = time - ellipse_center
    
    # Calculate the semi-ellipse shape using Theano operations
    ellipse_value = b * tt.sqrt(1 - (x_ellipse / a)**2)
    
    # Only apply the semi-ellipse where x is within [start_time, end_time]
    spot = tt.switch((time >= start_time) & (time <= end_time), ellipse_value, y)

    return spot

def spot_model_gaussian(a, b, time, gaussian_center):

    # Create a flat line of zeros
    y = tt.zeros((len(time),))

    # Calculate the Gaussian values
    gaussian_value = a * tt.exp(-0.5 * ((time - gaussian_center) / b)**2)
    
    # Ensure the Gaussian is applied across the entire array
    spot = tt.switch(gaussian_value > 0, gaussian_value, y)

    return spot

def light_model(priors, filters, orbit, time, ramp_type):
     ramp = ramp_scale(priors, filters, time, ramp_type)
     
     u = [priors[filters]['u1'], priors[filters]['u2']]#pm.math.concatenate([priors[filters]['u1'], priors[filters]['u2']])
     
     light_curves = (xo.LimbDarkLightCurve(u).get_light_curve(orbit=orbit, r=priors[filters]['RP_RS'], t=time[filters]) + 1)
     
     if star_spot_triangle == True:
         spot =  spot_model_triangle(priors[filters]['spot_a'], priors[filters]['spot_b'], time[filters], priors[filters]['spot_center'])
     elif star_spot_ellipse == True:
         spot = spot_model_ellipse(priors[filters]['spot_a'], priors[filters]['spot_b'], time[filters], priors[filters]['spot_center'])
     elif star_spot_gaussian == True:
         spot = spot_model_gaussian(priors[filters]['spot_a'], priors[filters]['spot_b'], time[filters], priors[filters]['spot_center'])
     else:
         spot=0
         
     return light_curves, ramp, spot


i=0

priors, priors_to_apply, fit_bounds, filenames, errorfile, wavelength_range, num_of_channels, norm_length = initialize.initialize(priors_to_apply, fit_bounds, planet, inst_to_fit, wavelength_range, broadband, data_extract, make_figures)


priors[filters]['u1'], priors[filters]['u2'], priors[filters]['err_u1'], priors[filters]['err_u2'] = gld.get_limb_darkening(broadband, filters, wavelength_range, i)
fit_bounds[filters]['sigma']['u1'] =  priors[filters]['err_u1']
fit_bounds[filters]['sigma']['u2'] =  priors[filters]['err_u2']
fit_bounds[filters]['mean']['u1'] =  priors[filters]['u1']
fit_bounds[filters]['mean']['u2'] =  priors[filters]['u2']

time, data, sigma = gd.get_data(inst_to_fit, planet, make_CF, i, filenames, wavelength_range, errorfile, norm_length, broadband, divide_white, visit_to_fit)
priors = priors.to_dict()

filename_triangle = './fit_variables/shortwave444/fit_variables_table_triangle.txt'

filename_ellipse = './fit_variables/shortwave444/fit_variables_table_ellipse.txt'

filename_gaussian = './fit_variables/shortwave444/fit_variables_table_gaussian.txt'


fit_triangle = pd.read_csv(filename_triangle, delim_whitespace=True)

fit_ellipse = pd.read_csv(filename_ellipse, delim_whitespace=True)

fit_gaussian = pd.read_csv(filename_gaussian, delim_whitespace=True)


for variables in priors_to_apply[filters]:
    
    if star_spot_triangle==True:
        priors[filters][variables] = fit_triangle[filters][variables]
    elif star_spot_ellipse==True:
        priors[filters][variables] = fit_ellipse[filters][variables]
    elif star_spot_gaussian==True:
        priors[filters][variables] = fit_gaussian[filters][variables]


orbit = xo.orbits.KeplerianOrbit(period=priors[filters]['period'], t0=priors[filters]['t0'], a=priors[filters]['A'], incl=priors[filters]['inc'], omega=(90*(np.pi/180)), ecc=0.0)

model, ramp, spot = light_model(priors, filters, orbit, time, ramp_type[filters])

model = (np.reshape(model.eval(), [len(time[filters]),])*ramp) + spot.eval()
residuals = (data[filters]+spot.eval()) - model

fig, ax = plt.subplots(2,1, figsize=[20,20])
fontsize=25

ax[0].errorbar(time[filters], data[filters], sigma[filters],fmt='.', color = 'blue', markersize = 20.0, alpha = 0.2)

ax[0].plot(time[filters], model, '--r', linewidth = 4.0, zorder=10)

ax[1].scatter(time[filters], residuals, marker='o', color='black', alpha = 1.0)

ax[0].set_ylabel('Normalized Flux', fontsize=fontsize)
ax[1].set_ylabel('Residuals', fontsize=fontsize)
ax[1].set_xlabel('time (days)', fontsize=fontsize)

ax[0].tick_params(axis='x', labelsize=fontsize)
ax[0].tick_params(axis='y', labelsize=fontsize)

ax[1].tick_params(axis='x', labelsize=fontsize)
ax[1].tick_params(axis='y', labelsize=fontsize)

ax[0].set_ylim(0.968, 0.971)
ax[0].set_xlim(fit_triangle[filters]['spot_center']-0.03, fit_triangle[filters]['spot_center']+0.03)
ax[1].set_xlim(fit_triangle[filters]['spot_center']-0.03, fit_triangle[filters]['spot_center']+0.03)

ax[1].plot(time[filters], spot.eval(), color='red', linewidth=5)
#
t = ax[0].xaxis.get_offset_text()
t.set_size(fontsize)

t = ax[1].xaxis.get_offset_text()
t.set_size(fontsize)

ax[0].set_title('F210W Broadband light curve (occulted star spot)', fontsize=fontsize)

if star_spot_triangle==True:
    plt.savefig('shortwave_triangle_spot.png', dpi=300)
elif star_spot_ellipse==True:
    plt.savefig('shortwave_ellipse_spot.png', dpi=300)
elif star_spot_gaussian==True:
    plt.savefig('./spot_heights/shortwave_gaussian_spot.png', dpi=300)    

    
    

