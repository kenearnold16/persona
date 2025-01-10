import numpy as np 
import pull_data as pull
import pdb
import pandas as pd
import find_visits
import exoplanet as xo

def HST_detrend(time, tau, hooks):
      return 1 - hooks*np.exp(-(np.array(time))/tau)

def light_model(pars, orbit, time, ramp_type):
    
     ramp = ramp_scale(pars, time, ramp_type)
     
     u = [pars['u1'], pars['u2']]
     
     light_curves = (xo.LimbDarkLightCurve(u).get_light_curve(orbit=orbit, r=pars['RP_RS'], t=time) + 1)
     
     return light_curves, ramp

def ramp_scale(pars, time, ramp_type):
    
     if ramp_type == 'line':
         median_time = np.median(time)
         ramp = 1 + pars['ramp1'] + pars['slope']*(time - median_time)
     
     elif ramp_type == 'exponential':
         median_time = np.median(time)
         t = (time - median_time)
         ramp = 1 + pars['ramp1']*np.exp(-(t)/pars['tau']) +  pars['slope']*t
         
     elif ramp_type == 'polynomial':
         median_time = np.median(time)
         ramp = 1 + pars['ramp1'] + pars['ramp2']*(time - median_time) + pars['slope']*(time - median_time)**2
     
     elif ramp_type == 'none':
         ramp = 1.0
     
     elif ramp_type == 'hooks':
         
         hst_hook = []
         t_visit, segments = find_visits.find_visits(time)
         median_time = np.median(time)
         
         for i in range(-1, len(t_visit)-1):
             i+=1 
   
             t_begin = t_visit[i]
            
             start = int(segments[i][0])
             end = int(segments[i][1])
            
             time1 = time[start:end] - t_begin
             if i< 1:
                 hst_hook.append(HST_detrend(time1, pars['tau1'], pars['hooks1']))
             elif i >= 1:
                 hst_hook.append(HST_detrend(time1, pars['tau2'], pars['hooks2']))
             #elif i >=2:
               #  hst_hook.append(HST_detrend(time1, pars['tau3'], pars['hooks3']))


         hst_hook = np.concatenate(hst_hook)
         
         median_time = np.median(time)
         
         #ramp = 1 + pars['ramp1'] + pars['ramp2']*(time - median_time) +  pars['slope']*(time - median_time)**2
         ramp = 1 + pars['ramp1'] + pars['slope']*(time - median_time)# +  pars['slope']*(time - median_time)**2

         ramp = hst_hook*ramp
         
     return ramp      

def get_data(inst_to_fit, planet, make_CF, i, filenames, wavelength_range, errorfile, norm_length, broadband, divide_white, visit_to_fit):
    
    data={}
    time={}
    sigma={}
    
    if 'MIRI' in inst_to_fit:
        time['MIRI'], data['MIRI'], sigma['MIRI'], wavelength = pull.pull_data_h5(filenames['MIRI'], i, make_CF, 'MIRI')
        wavelength_range['MIRI'][i]=wavelength[i]
        
        data['MIRI'] = data['MIRI'][200:len(data['MIRI'])]
        time['MIRI'] = time['MIRI'][200:len(time['MIRI'])]
        sigma['MIRI'] = sigma['MIRI'][200:len(sigma['MIRI'])]
        
        remove1=2550
        remove2=2900
        
        data['MIRI'] = np.concatenate([data['MIRI'][0:remove1], data['MIRI'][remove2:len(data['MIRI'])-1]])
        time['MIRI'] = np.concatenate([time['MIRI'][0:remove1], time['MIRI'][remove2:len(time['MIRI'])-1]])
        sigma['MIRI'] = np.concatenate([sigma['MIRI'][0:remove1], sigma['MIRI'][remove2:len(sigma['MIRI'])-1]])
        
        #time['MIRI'] = time['MIRI'] + 0.00005
    
    if 'NIRCam444' in inst_to_fit:
        #time['NIRCam444'], data['NIRCam444'], sigma['NIRCam444'], wavelength = pull.pull_data_h5(filenames['NIRCam444'], i, make_CF, 'NIRCam444')
        time['NIRCam444'], data['NIRCam444'], sigma['NIRCam444'] = pull.pull_data_txt(filenames['NIRCam444'], errorfile['NIRCam444'], i, norm_length['NIRCam444'], make_CF, 'NIRCam444')
        
        #find_zero=False
        #Tc = 56486.925
        #P = 3.067851921
       
        #while find_zero == False:
        #    i+=1
        #    check = Tc + P*i
            
           # if check < time['NIRCam444'][len(time['NIRCam444'])-1] and check > time['NIRCam444'][0]:
           #     j=0
           #     while (time['NIRCam444'][j] - check)*360 < 1.9372318548266776:
           #         j+=1
           #         if ((time['NIRCam444'][j] - check)*360 < 1.9372318548266776) == False:
           #             pdb.set_trace()
        remove1=372
        remove2=445
            #
        #data['NIRCam444'] = np.concatenate([data['NIRCam444'][0:remove1], data['NIRCam444'][remove2:len(data['NIRCam444'])-1]])
        #time['NIRCam444'] = np.concatenate([time['NIRCam444'][0:remove1], time['NIRCam444'][remove2:len(time['NIRCam444'])-1]])
        #sigma['NIRCam444'] = np.concatenate([sigma['NIRCam444'][0:remove1], sigma['NIRCam444'][remove2:len(sigma['NIRCam444'])-1]])

        
    if 'NIRCam322' in inst_to_fit:
        #time['NIRCam322'], data['NIRCam322'], sigma['NIRCam322'], wavelength = pull.pull_data_h5(filenames['NIRCam322'], i, make_CF, 'NIRCam322')
        time['NIRCam322'], data['NIRCam322'], sigma['NIRCam322'] = pull.pull_data_txt(filenames['NIRCam322'], errorfile['NIRCam322'], i, norm_length['NIRCam322'], make_CF, 'NIRCam322')

        #find_zero=False
        #Tc = 56486.925
        #P = 3.067851921
        #remove1 = 708
        #remove2 = 848
        
        
#        data['NIRCam322'] = np.concatenate([data['NIRCam322'][0:remove1], data['NIRCam322'][remove2:len(data['NIRCam322'])-1]])
#        time['NIRCam322'] = np.concatenate([time['NIRCam322'][0:remove1], time['NIRCam322'][remove2:len(time['NIRCam322'])-1]])
#        sigma['NIRCam322'] = np.concatenate([sigma['NIRCam322'][0:remove1], sigma['NIRCam322'][remove2:len(sigma['NIRCam322'])-1]])
        
  #      while find_zero == False:
   #         i+=1
    #        check = Tc + P*i
     #       
      #      if check < time['NIRCam322'][len(time['NIRCam322'])-1] and check > time['NIRCam322'][0]:
       #         j=0
        #        while (time['NIRCam322'][j] - check)*360 < 12.183130984485615:
         #           j+=1
          #          if ((time['NIRCam322'][j] - check)*360 < 12.183130984485615) == False:
           #             pdb.set_trace()
           
            
    if 'WFC3' in inst_to_fit:
                
        if broadband == True:
            filenames['WFC3'] = './DataAnalysis/WFC3/' + planet + '/lc_white.txt'
        else:
            filenames['WFC3'] = './DataAnalysis/WFC3/' + planet + '/speclc' + str(wavelength_range['WFC3'][i]) + '.txt'

        time['WFC3'], data['WFC3'], sigma['WFC3'] = pull.pull_data_hst(filenames['WFC3'], i, make_CF, 'WFC3', visit_to_fit)
        
        if divide_white == True:
          
            filename = './fit_variables/WFC3/fit_variables_table_BB.txt'
    
            fit = pd.read_csv(filename, delim_whitespace=True)
            pars = fit['WFC3']
            
            priors = pd.read_csv('./priors/priors_' + planet + '.csv', sep=',', index_col=[0], header=[0])
            
            orbit = xo.orbits.KeplerianOrbit(period=priors['WFC3']['period'], t0=priors['WFC3']['t0'], a=priors['WFC3']['A'], incl=priors['WFC3']['inc']*(np.pi/180), omega=(90*(np.pi/180)), ecc=0.0)
    
            light_curve, ramp = light_model(pars, orbit, time['WFC3'], 'hooks')
            
            light_curve = np.reshape(light_curve.eval(), [len(time['WFC3']),])

            model = (light_curve*ramp)
            
            data['WFC3'] = data['WFC3']/model
        
    if 'shortwave444' in inst_to_fit:
        
        time['shortwave444'], data['shortwave444'], sigma['shortwave444'] = pull.pull_data_short(filenames['shortwave444'], norm_length['shortwave444'], make_CF, 'shortwave444')
            
        sigma['shortwave444'] = sigma['shortwave444']*1.0e8
        
        data['shortwave444'] = np.reshape(data['shortwave444'], [len(time['shortwave444']),])
        
        sigma['shortwave444'] = np.reshape(sigma['shortwave444'], [len(time['shortwave444']),])
        
        
        #find_zero=False
        #Tc = 56486.925
        #P = 3.067851921
        #while find_zero == False:
       #     i+=1
       #     check = Tc + P*i
       #     
       ##     if check < time['shortwave444'][len(time['shortwave444'])-1] and check > time['shortwave444'][0]:
       ##         find_zero=True
        #        print((time['shortwave444'][372] - check)*360)
        #       print((time['shortwave444'][445] - check)*360)#

        remove1=372
        remove2=445
            #
        #data['shortwave444'] = np.concatenate([data['shortwave444'][0:remove1], data['shortwave444'][remove2:len(data['shortwave444'])-1]])
        #time['shortwave444'] = np.concatenate([time['shortwave444'][0:remove1], time['shortwave444'][remove2:len(time['shortwave444'])-1]])
        #sigma['shortwave444'] = np.concatenate([sigma['shortwave444'][0:remove1], sigma['shortwave444'][remove2:len(sigma['shortwave444'])-1]])
        
        data['shortwave444'][133:360] = data['shortwave444'][133:360] + 0.00047941339150863295
        data['shortwave444'][360:len(data['shortwave444'])] = data['shortwave444'][360:len(data['shortwave444'])] + 0.00025834568980198005
            
    if 'shortwave322' in inst_to_fit:
        
        time['shortwave322'], data['shortwave322'], sigma['shortwave322'] = pull.pull_data_short(filenames['shortwave322'], norm_length['shortwave322'], make_CF, 'shortwave322')
        sigma['shortwave322'] =  sigma['shortwave322']*1.0e8
        
        data['shortwave322'][0] = data['shortwave322'][1]
        data['shortwave322'][133] = data['shortwave322'][134]
        data['shortwave322'][256] = data['shortwave322'][257]
    
        data['shortwave322'] = np.reshape(data['shortwave322'], [len(time['shortwave322']),])
        
        sigma['shortwave322'] = np.reshape(sigma['shortwave322'], [len(time['shortwave322']),])
        
        #time['shortwave322'] = time['NIRCam322']
        
        
    return time, data, sigma