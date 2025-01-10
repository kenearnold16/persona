import numpy as np
import pymc3 as pm
import pymc3_ext as pmx
import exoplanet as xo
import pdb
import matplotlib.pyplot as plt
import arviz as az
import corner
import rename_frame as rf
import write_values_to_txt as wvtt
import plot_LC
import get_data as gd
import find_visits
import initialize 
import get_limb_darkening as gld
import sys
import pandas as pd
import theano.tensor as tt

planet = 'wasp80b'

broadband=False
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
make_figures['save_LC_plots'] = False
make_figures['show_fit_model'] = True
make_figures['DataReduction'] = False
make_figures['corner_plots'] = False
make_figures['Rolling_BKG'] = False

inst_to_fit = ['NIRCam444'] #['joint', 'NIRCam322','NIRCam444', 'MIRI', 'hst', 'shortwave322', 'shortwave444']

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

if broadband == True:
    ramp_type['WFC3'] = 'hooks'
elif broadband == False and divide_white == True:
    ramp_type['WFC3'] = 'line'
else:
    ramp_type['WFC3'] = 'hooks'

ramp_type['shortwave444'] = 'polynomial'
ramp_type['shortwave322'] = 'polynomial'

priors, priors_to_apply, fit_bounds, filenames, errorfile, wavelength_range, num_of_channels, norm_length = initialize.initialize(priors_to_apply, fit_bounds, planet, inst_to_fit, wavelength_range, broadband, data_extract, make_figures)


transit_depth = np.zeros(num_of_channels)
TD_error1 = np.zeros(num_of_channels)
TD_error2 = np.zeros(num_of_channels)

correction_factor_array = np.zeros(num_of_channels)
rstar = 0.586
np.random.seed(4637286)

def HST_detrend(time, tau, hooks):
      return 1 - hooks*np.exp(-(np.array(time))/tau)
 
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
     
     elif ramp_type == 'none':
         ramp = 1.0
     
     elif ramp_type == 'hooks':
         hst_hook = []
         
         median_time = np.median(time[filters])
         
         t_visit, segments = find_visits.find_visits(time[filters])
        
         for i in range(-1, len(t_visit)-1):
             i+=1 
   
             t_begin = t_visit[i]
            
             start = int(segments[i][0])
             end = int(segments[i][1])
            
             time1 = time[filters][start:end] - t_begin
             if i < 1:
                 hst_hook.append(HST_detrend(time1, priors[filters]['tau1'], priors[filters]['hooks1']))
             elif i >= 1:
                 hst_hook.append(HST_detrend(time1, priors[filters]['tau2'], priors[filters]['hooks2']))
             #elif i >= 2:
               #  hst_hook.append(HST_detrend(time1, priors[filters]['tau3'], priors[filters]['hooks3']))

                 
         hst_hook = pm.math.concatenate(hst_hook)
         
         median_time = np.median(time[filters])
         
         #ramp = 1 + priors[filters]['ramp1'] + priors[filters]['ramp2']*(time[filters] - median_time) +  priors[filters]['slope']*(time[filters] - median_time)**2
         ramp = 1 + priors[filters]['ramp1'] + priors[filters]['slope']*(time[filters] - median_time)# +  priors[filters]['slope']*(time[filters] - median_time)**2

         ramp = hst_hook*ramp
         
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
    gaussian_value = b * tt.exp(-0.5 * ((time - gaussian_center) / a)**2)
    
    # Ensure the Gaussian is applied across the entire array
    spot = tt.switch(gaussian_value > 0, gaussian_value, y)

    return spot

def light_model(priors, filters, orbit, time, ramp_type):
     ramp = ramp_scale(priors, filters, time, ramp_type)
     
     u = [priors[filters]['u1'], priors[filters]['u2']]
     #u = pm.math.concatenate([priors[filters]['u1'], priors[filters]['u2']])
     
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

priorslist = []
labels = [] 
                
priors = priors.to_dict()

for i in range(-1, num_of_channels-1):
    i+=1
    
    for filters in inst_to_fit:
        if filters == 'joint':
            pass
        else:
            priors[filters]['u1'], priors[filters]['u2'], priors[filters]['err_u1'], priors[filters]['err_u2'] = gld.get_limb_darkening(broadband, filters, wavelength_range, i)
            fit_bounds[filters]['sigma']['u1'] =  priors[filters]['err_u1']
            fit_bounds[filters]['sigma']['u2'] =  priors[filters]['err_u2']
            fit_bounds[filters]['mean']['u1'] =  priors[filters]['u1']
            fit_bounds[filters]['mean']['u2'] =  priors[filters]['u2']

            
    time, data, sigma = gd.get_data(inst_to_fit, planet, make_CF, i, filenames, wavelength_range, errorfile, norm_length, broadband, divide_white, visit_to_fit)

    with pm.Model() as model:
        
        for filters in inst_to_fit:
            for variables in priors_to_apply[filters]:

                if priors_to_apply[filters][variables]['distribution'] == 'Uniform':
                    
                    priors[filters][variables] = (pm.Uniform(variables + '_' + filters, lower=fit_bounds[filters]['lower'][variables], upper=fit_bounds[filters]['upper'][variables], shape=1))#, testval=priors[filters][variables]))

                    labels.append(variables + '_' + filters)                    

                elif priors_to_apply[filters][variables]['distribution'] == 'Normal':

                    priors[filters][variables] = (pm.Normal(variables, mu=fit_bounds[filters]['mean'][variables], sd=fit_bounds[filters]['sigma'][variables], shape=1))#, testval=priors[filters][variables][0]))
                        
                    labels.append(variables)
                
            if ('joint' in inst_to_fit) and ('cos_i' in priors['joint']) or ('joint' in inst_to_fit) and ('log_A' in priors['joint']):
                
                priors['joint']['inc'] = pm.math.tt.arccos(priors['joint']['cos_i'])
                priors['joint']['A'] = pm.math.tt.pow(10,priors['joint']['log_A'])
                orbit = xo.orbits.KeplerianOrbit(period=priors['joint']['period'], t0=priors['joint']['t0'], a=priors['joint']['A'], incl=priors['joint']['inc'], omega=(90*(np.pi/180)), ecc=0.0)

            else:
                
                if 'cos_i' in priors_to_apply[filters] or 'log_A' in priors_to_apply[filters]:
                    priors[filters]['inc'] = pm.math.tt.arccos(priors[filters]['cos_i'])
                    priors[filters]['A'] = pm.math.tt.pow(10,priors[filters]['log_A'])
                
                orbit = xo.orbits.KeplerianOrbit(period=priors[filters]['period'], t0=priors[filters]['t0'], a=priors[filters]['A'], incl=priors[filters]['inc'], omega=(90*(np.pi/180)), ecc=0.0)
                
            try:
                
                light_curves, ramp, spot = light_model(priors, filters, orbit, time, ramp_type[filters])
                
                #pdb.set_trace()
                light_curve = ((pm.math.sum(light_curves, axis=-1))*(ramp)) + spot
                
                pm.Normal("obs_" + filters, mu=light_curve, sd=sigma[filters], observed=data[filters])

            except:
                if filters == 'joint':
                    pass
                else:
                    raise AttributeError("FAILED TO BUILD MODEL")
                    

        map_soln = pmx.optimize(start=model.test_point)
        
    with model:
        trace = pmx.sample(
            tune=10000,
            draws=20000,
            start=map_soln,
            cores=4,
            chains=4,
            target_accept=0.95,
            return_inferencedata=True)
    
    az.plot_trace(trace)
    
    channel = str(i)
    
    if 'joint' in inst_to_fit:
        plt.savefig('./figs/joint_fit/samples/samples_' + planet + '_' + channel + '.png')
    else:
        plt.savefig('./figs/' + inst_to_fit[0] + '/samples/samples_' + planet + '_' + channel + '.png')
    
    for filters in priors:
        for variables in priors[filters]:
            priorslist.append(priors[filters][variables])
    
    if make_figures['corner_plots'] == True:
        
        truth = dict(
            zip(
                labels,
                pmx.eval_in_model(priorslist, model.test_point, model=model),
            )
            
        )
        
        _ = corner.corner(
            trace,
            var_names=labels,
            truths=truth,
        )
        
        if 'joint' in inst_to_fit:
            plt.savefig('./figs/joint_fit/corner_plots/corner_plot_' + planet + '_' + channel + '.png')
        else:
            plt.savefig('./figs/' + inst_to_fit[0] + '/corner_plots/corner_plot_' + planet + '_' + channel + '.png')

    mcmc = trace.posterior.quantile((.16, .5, .84), dim=("chain", "draw"))
    
    fit_variables = {}
    fit_error1 = {}
    fit_error2 = {}
    
    fit_variables_set = {}
    fit_error1_set = {}
    fit_error2_set = {}
    
    for filters in inst_to_fit:
        fit_variables = {}
        fit_error1 = {}
        fit_error2 = {}
        for variables in mcmc:
            
            fit = np.float64(mcmc[variables][1])
            
            fit_variables[variables] = fit 
            
            fit_error1[variables] = np.float64(mcmc[variables][1]) - np.float64(mcmc[variables][0])
            fit_error2[variables] = np.float64(mcmc[variables][2]) - np.float64(mcmc[variables][1])
            
            if variables == 'RP_RS_' + filters:
                variables = 'RP_RS^2'
                fit_error1[variables] = np.float64(mcmc['RP_RS_' + filters][1])**2 - np.float64(mcmc['RP_RS_' + filters][0])**2
                fit_error2[variables] = np.float64(mcmc['RP_RS_' + filters][2])**2 - np.float64(mcmc['RP_RS_' + filters][1])**2
                fit = np.float64(mcmc['RP_RS_' + filters][1]**2)
            else:
                pass
            
            mcmc = trace.posterior.quantile((.16, .5, .84), dim=("chain", "draw"))
                        
            fit_variables[variables] = fit
        
            #fit_error1[variables] = mcmc[1] - mcmc[0]
            #fit_error2[variables] = mcmc[2] - mcmc[1]
            #
            
        fit_variables_set[filters] = fit_variables
        fit_error1_set[filters] = fit_error1
        fit_error2_set[filters] = fit_error2
        
    fit_variables = {}
    fit_variables = fit_variables_set
    fit_error1 = {}
    fit_error1 = fit_error1_set
    fit_error2 = {}
    fit_error2 = fit_error2_set
    
    model={}
    try:
        if 'u_joint' in fit_variables['joint']:
            for filters in inst_to_fit:
                fit_variables[filters]['u'] =  fit_variables['joint']['u_joint']
        if 'RP_RS_joint' in fit_variables['joint']:
            for filters in inst_to_fit:
                fit_variables[filters]['RP_RS'] =  fit_variables['joint']['RP_RS_joint']
    except:
        pass
    
    for filters in inst_to_fit:
        if filters == 'joint':
            pass
        else:
            fit_variables, fit_error1, fit_error2 = rf.rename_frame(inst_to_fit, fit_variables, fit_error1, fit_error2, priors, priors_to_apply)
            
            if 'joint' in inst_to_fit:
                
                fit_variables['joint']['inc'] = pm.math.tt.arccos(fit_variables['joint']['cos_i']).eval()
                fit_variables['joint']['A'] = pm.math.tt.pow(10,fit_variables['joint']['log_A']).eval()
                
                orbit = xo.orbits.KeplerianOrbit(period=fit_variables['joint']['period'], t0=fit_variables['joint']['t0'], a=fit_variables['joint']['A'], incl=fit_variables['joint']['inc'], omega=(90*(np.pi/180)), ecc=0.0)
             
            elif ('joint' not in priors_to_apply[filters] and 'cos_i' in fit_variables[filters]) or  ('joint' not in priors_to_apply[filters] and 'log_A' in fit_variables[filters]):
                
                fit_variables[filters]['inc'] = (pm.math.tt.arccos(fit_variables[filters]['cos_i']).eval())
                fit_variables[filters]['A'] = (pm.math.tt.pow(10,fit_variables[filters]['log_A']).eval())
                
                orbit = xo.orbits.KeplerianOrbit(period=fit_variables[filters]['period'], t0=fit_variables[filters]['t0'], a=fit_variables[filters]['A'], incl=fit_variables[filters]['inc'], omega=(90*(np.pi/180)), ecc=0.0)
                 
            elif 'cos_i' not in fit_variables[filters] or 'log_A' not in fit_variables[filters] :
                
                priors[filters]['inc'] = [pm.math.tt.arccos(priors[filters]['cos_i'])]
                priors[filters]['A'] = [pm.math.tt.pow(10,priors[filters]['log_A'])]
                
                orbit = xo.orbits.KeplerianOrbit(period=priors[filters]['period'], t0=priors[filters]['t0'], a=priors[filters]['A'], incl=priors[filters]['inc'], omega=(90*(np.pi/180)), ecc=0.0)
                
            #fit_variables = rf.rename_frame(inst_to_fit, fit_variables, priors, priors_to_apply)
            
            if 'u1' in fit_variables[filters]:
                fit_variables[filters]['u1'] = [(fit_variables[filters]['u1'])]
                fit_variables[filters]['u2'] = [(fit_variables[filters]['u2'])]
            
            for variables in priors_to_apply[filters]:
                priors[filters][variables] = fit_variables[filters][variables]
            
            model[filters], ramp, spot = light_model(priors, filters, orbit, time, ramp_type[filters])
                    
            if (filters == 'WFC3') and broadband==True:
                ramp = np.reshape(ramp.eval(), [len(time[filters]),])
                
            model[filters] = (np.reshape(model[filters].eval(), [len(time[filters]),])*ramp)#+ spot.eval()
            residuals = data[filters] - model[filters]

            if make_figures['show_fit_model'] == True:

                fig, ax = plt.subplots(2,1, figsize=[20,20])
                fontsize=20 
    
                ax[0].errorbar(time[filters], data[filters], sigma[filters],fmt='.', color = 'blue', markersize = 20.0, alpha = 0.2)
                
                ax[0].plot(time[filters], model[filters], '--r', linewidth = 4.0, zorder=10)
                
                
                ax[1].scatter(time[filters], residuals, marker='o', color='black', alpha = 1.0)
                
                
                ax[0].set_ylabel('Normalized Flux', fontsize=fontsize)
                ax[1].set_ylabel('Residuals', fontsize=fontsize)
                ax[1].set_xlabel('time (days)', fontsize=fontsize)
            
                ax[0].tick_params(axis='x', labelsize=fontsize)
                ax[0].tick_params(axis='y', labelsize=fontsize)
            
                ax[1].tick_params(axis='x', labelsize=fontsize)
                ax[1].tick_params(axis='y', labelsize=fontsize)
                
                #ax[0].set_ylim(0.968, 0.97)
                #ax[1].set_xlim(priors[filters]['spot_center']-0.03, priors[filters]['spot_center']+0.03)
                ax[1].set_ylim(-0.001, 0.001)

                #ax[1].plot(time[filters], spot.eval(), color='red', linewidth=5)

                #ax[0].set_xlim(time[filters][300], time[filters][450])
                #ax[1].set_xlim(time[filters][300], time[filters][450])
                
                #
                t = ax[0].xaxis.get_offset_text()
                t.set_size(fontsize)
            
                t = ax[1].xaxis.get_offset_text()
                t.set_size(fontsize)
                
                    
    if make_figures['save_LC_plots'] == True:
        plot_LC.plot_light_curve(time, data, sigma, model, channel, inst_to_fit, planet)
        plt.show()

     
    if 'joint' in inst_to_fit:
        wvtt.write_values_to_txt(fit_variables, './fit_variables/joint_fit/fit_variables_table_' + channel + '.txt')
        wvtt.write_values_to_txt(fit_error1, './fit_variables/joint_fit/fit_error1_table_' + channel + '.txt')
        wvtt.write_values_to_txt(fit_error2, './fit_variables/joint_fit/fit_error2_table_' + channel + '.txt')
    else:
        for filters in inst_to_fit:
            if broadband==True:
                wvtt.write_values_to_txt(fit_variables, './fit_variables/' + filters + '/fit_variables_table_BB.txt')
                wvtt.write_values_to_txt(fit_error1, './fit_variables/'  + filters +  '/fit_error1_table_BB.txt')
                wvtt.write_values_to_txt(fit_error2, './fit_variables/' + filters +  '/fit_error2_table_BB.txt')
            else:
                wvtt.write_values_to_txt(fit_variables, './fit_variables/' + filters + '/fit_variables_table_' + channel + '.txt')
                wvtt.write_values_to_txt(fit_error1, './fit_variables/'  + filters +  '/fit_error1_table_' + channel + '.txt')
                wvtt.write_values_to_txt(fit_error2, './fit_variables/' + filters +  '/fit_error2_table_' + channel + '.txt')
                
    if 'joint' not in inst_to_fit:
        residuals = data[inst_to_fit[0]]- model[inst_to_fit[0]]
        
        std_residuals = np.std(residuals)
        mean_error = np.mean(sigma[inst_to_fit[0]])
    
        correction_factor_array[i] = std_residuals/mean_error
    
        transit_depth[i] = fit_variables[inst_to_fit[0]]['RP_RS^2']
        TD_error1[i] = fit_error1[inst_to_fit[0]]['RP_RS^2']
        TD_error2[i] = fit_error2[inst_to_fit[0]]['RP_RS^2']
        
if make_CF == True:
    wvtt.write_values_to_txt(correction_factor_array, './correction_factor/correction_factor_' + inst_to_fit[0] + '.txt')
else:
    pass
    
if broadband == False:
    transmission={}
    transmission['transit depth']=transit_depth
    transmission['err1']=TD_error1
    transmission['err2']=TD_error2
    transmission['wavelength']=np.float64(wavelength_range[inst_to_fit[0]])
    transmission['bin_width'] = np.ones(len(wavelength_range[inst_to_fit[0]]))*(0.015/2)
    if inst_to_fit[0] == 'NIRCam322' or inst_to_fit[0] == 'NIRCam444':
        wvtt.write_values_to_txt(transmission, './transmission_spectra/jwst/'+ inst_to_fit[0] + '/transmission_spectra_' + inst_to_fit[0] + '.txt')
    elif inst_to_fit[0] == 'WFC3':
        wvtt.write_values_to_txt(transmission, './transmission_spectra/'+ inst_to_fit[0] + '/transmission_spectra_' + inst_to_fit[0] + '.txt')

        