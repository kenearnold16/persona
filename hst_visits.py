
def hst_visits(time):
    
    t_visit = [[time['hst'][0], time['hst'][19], time['hst'][38]]] 
    segments = [[[0,19], [19, 38], [38, 57]]]
              
    return t_visit, segments