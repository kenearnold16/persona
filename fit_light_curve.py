import matplotlib.pyplot as plt
import numpy as np
import batman
import emcee
import corner
import pdb
import get_limb_darkening as gld
import hst_visits

rstar = 0.586

def plot_light_curve(time, data, sigma, model, channel, filters):
    
    fontsize = 35
    plt.figure()
    i = -1
    fig, ax = plt.subplots(2,len(data), figsize=[20,20])

    for names in data:
        i += 1
        time[names] = time[names] + 2400000.5
        
        try:

            ax[0,i].errorbar(time[names], data[names], sigma[names],fmt='.', color = 'blue', markersize = 20.0, alpha = 0.2)
            ax[0,i].plot(time[names], model[names], '--r', linewidth = 4.0, zorder=10)
        
            residuals = data[names] - model[names]
        
            ax[1,i].scatter(time[names], residuals, marker='o', color='black', alpha = 1.0)
        
            ax[0,0].set_ylabel('Normalized Flux', fontsize=fontsize)
            ax[1,0].set_ylabel('Residuals', fontsize=fontsize)
            ax[1,0].set_xlabel('time (days)', fontsize=fontsize)
        
            ax[0,i].tick_params(axis='x', labelsize=fontsize)
            ax[0,i].tick_params(axis='y', labelsize=fontsize)
        
            ax[1,i].tick_params(axis='x', labelsize=fontsize)
            ax[1,i].tick_params(axis='y', labelsize=fontsize)
        
            t = ax[0,i].xaxis.get_offset_text()
            t.set_size(fontsize)
        
            t = ax[1,i].xaxis.get_offset_text()
            t.set_size(fontsize)
            
        except:
            
            ax[0].errorbar(time[names], data[names], sigma[names],fmt='.', color = 'blue', markersize = 20.0, alpha = 0.2)
            ax[0].plot(time[names], model[names], '--r', linewidth = 4.0, zorder=10)
        
            residuals = data[names] - model[names]
        
            ax[1].scatter(time[names], residuals, marker='o', color='black', alpha = 1.0)
        
            ax[0].set_ylabel('Normalized Flux', fontsize=fontsize)
            ax[1].set_ylabel('Residuals', fontsize=fontsize)
            ax[1].set_xlabel('time (days)', fontsize=fontsize)
        
            ax[0].tick_params(axis='x', labelsize=fontsize)
            ax[0].tick_params(axis='y', labelsize=fontsize)
        
            ax[1].tick_params(axis='x', labelsize=fontsize)
            ax[1].tick_params(axis='y', labelsize=fontsize)
        
            t = ax[0].xaxis.get_offset_text()
            t.set_size(fontsize)
        
            t = ax[1].xaxis.get_offset_text()
            t.set_size(fontsize)
    
        fig.tight_layout()
        plt.savefig('./figs/' + filters + '/light_curves/wasp80b_light_curve_' + str(channel) + '.png', dpi=300)
        plt.show()

def ramp_scale(ramp1, slope, time):
    median_time = np.median(time)
    ramp = 1+ ramp1 + slope*(time - median_time)
    return ramp

def WFC3_detrend(t, tau, hooks):
     return 1 - hooks*np.exp(-(t)/tau)

def ramp_scale_quadratic(ramp1, ramp2, slope, time):
    median_time = np.median(time)
    ramp = 1 + ramp1 + ramp2*(time-median_time) + slope*(time-median_time)**2
    return ramp

def spot_model(t, priors):
    
    spot = np.sqrt((priors['spot_b'])**2 - (((priors['spot_b'])/(priors['spot_a']))**2)*(t - priors['spot_center'])**2)
    #spot1 = (t-priors['spot_center']+0.5*priors['spot_a'])*(2*priors['spot_b']/priors['spot_a'])
    #spot2 = (t-priors['spot_center']-0.5*priors['spot_a'])*(-2*priors['spot_b']/priors['spot_a'])
    #spot1[spot1<0] = 0.0
    #spot2[spot2<0]=0
    #spot1[spot1>priors['spot_b']]=0
    #spot2[spot2>priors['spot_b']]=0
    
    #spot = spot1 + spot2
    spot = np.nan_to_num(spot, copy=True, nan=0.0)
    
    return spot

def light_model(t, priors, inst_to_fit, ramp_type, star_spot):
    
    ECC = 0.0 # planetary eccentricity
    params = batman.TransitParams()
    
    params.per = priors['period']

    params.rp = priors['RP_RS']
    
    params.a = 10**priors['log_A']
    
    params.inc = np.arccos(priors['cos_i'])*(180/np.pi)
    
    params.t0 = priors['t0']
    
    params.limb_dark = "quadratic"   

    params.u = [priors['u1'], priors['u2']]

    if inst_to_fit == 'NIRCam444' or inst_to_fit == 'NIRCam322' or inst_to_fit == 'shortwave444' or inst_to_fit == 'shortwave322':
        if ramp_type[inst_to_fit] == 'line':
            ramp = ramp_scale(priors['ramp1'], priors['slope'], t)
        elif ramp_type[inst_to_fit] == 'polynomial':
            ramp = ramp_scale_quadratic(priors['ramp1'], priors['ramp2'], priors['slope'], t)
    
    elif inst_to_fit == 'WFC3':
        WFC3_hook = np.zeros(len(t))

        t_visit, segments = hst_visits.hst_visits(t)
        
        for i in range(-1, len(t_visit)-1):
            i+=1 
            t_begin = t_visit[i]
            
            start = int(segments[0][i][0])
            end = int(segments[0][i][1])
            
            time = t[start:end] - t_begin
            if i == 0:    
                WFC3_hook[start:end] = WFC3_detrend(time, priors['tau1'], priors['hooks1'])
            elif i > 0:
                WFC3_hook[start:end] = WFC3_detrend(time, priors['tau2'], priors['hooks2x'])

        
        WFC3_hook = np.array(WFC3_hook)
        
        ramp = ramp_scale(priors['ramp1'], priors['slope'], t)
        ramp = (WFC3_hook*ramp)
                                    
    params.ecc = ECC                      #eccentricity
    params.w = 90.                  #longitude of periastron (in degrees)

    if star_spot == True:
        spot = spot_model(t, priors)
    else: 
        spot=0
        
    m = batman.TransitModel(params, t)    #initializes model
        
    flux_model = (m.light_curve(params)*ramp) + spot

    return flux_model

def log_likelihood(priors, priors_to_apply, fit_bounds, time, data, sigma, ramp_type, star_spot):
        
    lnp_params = 0.0
    lnp_data = 0.0
            
    for names in data:
        
        model = light_model(time[names], priors[names], names, ramp_type, star_spot)
        lnp_data += -0.5 * np.sum(((data[names] - model)**2 / sigma[names]**2) + np.log(sigma[names]*2*np.pi))
        
        for variables in priors_to_apply[names]:
            
            if priors_to_apply[names][variables]['distribution'] == 'Uniform':
                
                if (priors[names][variables] < fit_bounds[names]['lower'][variables] or priors[names][variables] > fit_bounds[names]['upper'][variables]):
                    return -np.inf
                
            elif priors_to_apply[names][variables]['distribution'] == 'Normal':
                lnp_params += -0.5*((priors[names][variables] - priors[names][variables])**2 / priors[names]['err_' + variables]**2) + np.log(priors[names]['err_' + variables]*2*np.pi)

    return lnp_data + lnp_params

def MCMCWrapper(parsarray, priors, priors_to_apply, fit_bounds, time, data, sigma, labels, concordance, ramp_type, star_spot):
    for names in data:
        for variables in labels: 
            
            if variables in priors_to_apply[names]:
                priors[names][variables] = parsarray[concordance[names][variables]]

    lnp = log_likelihood(priors, priors_to_apply, fit_bounds, time, data, sigma, ramp_type, star_spot)
	
    return lnp

def fit_light_curve(parsarray, time, data, sigma, priors, priors_to_apply, fit_bounds, labels, concordance, channel, ramp_type, filters, star_spot):
    
    walkers = 100
        
    pos = parsarray + 1e-5 * np.random.randn(walkers, len(parsarray))
    
    for names in data:
        for variables in labels: 
           
            if variables in priors_to_apply[names]:
                priors[names][variables] = parsarray[concordance[names][variables]]

    nwalkers, ndim = pos.shape
        
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, MCMCWrapper, args=(priors, priors_to_apply, fit_bounds, time, data, sigma, labels, concordance, ramp_type, star_spot)
    )
    
    sampler.run_mcmc(pos, 2000, progress=True);
    
    fig, axes = plt.subplots(len(parsarray), figsize=(10,7), sharex=True)
    samples = sampler.get_chain()
            
    j = -1
    labels = []
    
    for filters in data:
        for names in priors_to_apply[filters]:
            j += 1
            ax = axes[j]
            ax.plot(samples[:, :, j], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(names + ' ' + str(filters))
            ax.yaxis.set_label_coords(-0.1, 0.5)
            labels.append(names)
            
        flat_samples = sampler.get_chain(discard=100, thin=10, flat=True)
        print(flat_samples.shape)
            
        plt.savefig('./figs/' + filters + '/samples/samples_wasp80b_' + channel + '.png')
        fig = corner.corner(
            flat_samples, labels=labels, truths=parsarray
        );
        

        plt.savefig('./figs/'  + filters + '/corner_plots/corner_plot_wasp80b_' + channel + '.png')

    #tau = sampler.get_autocorr_time()
    #print(tau)
    
    fit_variables = {}
    fit_error1 = {}
    fit_error2 = {}
    
    i=-1
    fit_variables_set = {}
    fit_error1_set = {}
    fit_error2_set = {}
    
    for names1 in data:
        fit_variables = {}
        fit_error1 = {}
        fit_error2 = {}
        for names2 in concordance[names1]:
            
            i+=1
    
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
                
            q = np.diff(mcmc)
            
            fit_variables[names2] = mcmc[1]
        
            fit_error1[names2] = mcmc[1] - mcmc[0]
            fit_error2[names2] = mcmc[2] - mcmc[1]
            
            if names2 == 'cos_i':
                flat_samples[:, i] = np.arccos(flat_samples[:, i])*(180/np.pi)   
                names2 = 'i'
            elif names2 == 'log_A':
                flat_samples[:, i] = 10**flat_samples[:, i]
                names2 = 'A'
            elif names2 == 'RP_RS':
                names2 = 'RP_RS^2'
                fit_error1[names2] = mcmc[1]**2 - mcmc[0]**2
                fit_error2[names2] = mcmc[2]**2 - mcmc[1]**2
                flat_samples[:, i] = flat_samples[:, i]**2
            else:
                pass
            
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            
            q = np.diff(mcmc)
            
            fit_variables[names2] = mcmc[1]
        
            #fit_error1[names2] = mcmc[1] - mcmc[0]
            #fit_error2[names2] = mcmc[2] - mcmc[1]
            #pdb.set_trace()
            
        fit_variables_set[names1] = fit_variables
        fit_error1_set[names1] = fit_error1
        fit_error2_set[names1] = fit_error2
        
    fit_variables = {}
    fit_variables = fit_variables_set
    fit_error1 = {}
    fit_error1 = fit_error1_set
    fit_error2 = {}
    fit_error2 = fit_error2_set
       
    model = {}
    
    for names in data:
        for variables in priors_to_apply[names]:
            priors[names][variables] = fit_variables[names][variables]
            
        model[names] = light_model(time[names], priors[names], names, ramp_type, star_spot)

    return fit_variables, fit_error1, fit_error2, model
