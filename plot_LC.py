import matplotlib.pyplot as plt
import pdb

def plot_light_curve(time, data, sigma, model, channel, inst_to_fit, planet):
    
    fontsize = 40
    plt.figure()
    i = -1
    
    if 'joint' in inst_to_fit:
        fig, ax = plt.subplots(2,len(inst_to_fit)-1, figsize=[90,20])
    else:
        fig, ax = plt.subplots(2,len(inst_to_fit), figsize=[25,20])
        
    for names in inst_to_fit:
        if names == 'joint':
            pass
        else:
            i += 1
            time[names] = time[names] + 2400000.5
            
            if len(inst_to_fit) > 1:
                
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
            
            elif len(inst_to_fit) == 1:
    
                ax[0].errorbar(time[names], data[names], sigma[names],fmt='.', color = 'blue', markersize = 20.0, alpha = 0.2)
                ax[0].plot(time[names], model[names], '--r', linewidth = 4.0, zorder=10)
            
                residuals = data[names] - model[names]
                ax[1].scatter(time[names], residuals, marker='o', color='black', alpha = 1.0)
            
                ax[0].set_ylabel('Normalized Flux', fontsize=fontsize)
                #ax[0].set_ylim([0.99, 1.01])
                ax[1].set_ylabel('Residuals', fontsize=fontsize)
                ax[1].set_xlabel('time (BJD)', fontsize=fontsize)
            
                ax[0].tick_params(axis='x', labelsize=fontsize)
                ax[0].tick_params(axis='y', labelsize=fontsize)
            
                ax[1].tick_params(axis='x', labelsize=fontsize)
                ax[1].tick_params(axis='y', labelsize=fontsize)
            
                t = ax[0].xaxis.get_offset_text()
                t.set_size(fontsize)
            
                t = ax[1].xaxis.get_offset_text()
                t.set_size(fontsize)
                
                #ax[0].set_title('NIRCam/F444W broadband light curve', fontsize=fontsize)
    
        fig.tight_layout()
        
        if 'joint' in inst_to_fit:
            pass
        else:
            plt.savefig('./figs/' + names + '/light_curves/' + planet + '_light_curve_' + channel + '.png', dpi=300)

    if 'joint' in inst_to_fit:
        plt.savefig('./figs/joint_fit/light_curves/' + planet + '_light_curve_' + channel + '.png', dpi=300)
