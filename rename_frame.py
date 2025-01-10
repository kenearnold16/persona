import pdb

def rename_frame(inst_to_fit, fit_variables, fit_error1, fit_error2, pars, priors_to_apply):
    #pdb.set_trace()
    for names in inst_to_fit:
        if names == 'joint':
            for labels in pars:
                for variables in pars[labels]:
                    try:
                        del fit_variables['joint'][variables + '_' + labels]
                        del fit_error1['joint'][variables + '_' + labels]
                        del fit_error2['joint'][variables + '_' + labels]

                    except:
                          pass
        else:
            for labels in pars:
                for variables in pars[labels]:
                    try:
                        #pdb.set_trace()
                        fit_variables[names][variables] = fit_variables[names][variables + '_' + names]
                        fit_error1[names][variables] = fit_error1[names][variables + '_' + names]
                        fit_error2[names][variables] = fit_error2[names][variables + '_' + names]

                    except:
                        pass
                    
                    try:
                        del fit_variables[names][variables + '_' + labels]
                        del fit_error1[names][variables + '_' + labels]
                        del fit_error2[names][variables + '_' + labels]

                    except:
                        pass
                    
            #pdb.set_trace()
            if 'joint' in inst_to_fit:
                for variables in priors_to_apply['joint']:
                    fit_variables[names][variables] = fit_variables['joint'][variables]
                    fit_error1[names][variables] = fit_error1['joint'][variables]
                    fit_error2[names][variables] = fit_error2['joint'][variables]

                if names == 'joint':
                    pass
                else:
                    for variables in priors_to_apply['joint']:
                        del fit_variables[names][variables]
                        del fit_error1[names][variables]
                        del fit_error2[names][variables]

    return fit_variables, fit_error1, fit_error2
 