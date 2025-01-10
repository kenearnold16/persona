import pdb

def find_visits(time):
    t_visit = []
    segments = []
    
    j=0
    k=0
    for i in range(-1, len(time)-1):
        try:
            delta_t = round(time[k+1],2) - round(time[k], 2)
        except:
            delta_t = time[k] - time[k-1]
            
        i+=1
        seg=[0,0]
        try:
            if round(time[i+1] - time[i],2) > round(delta_t, 2):
                j+=1
                
                t_visit.append(time[k])
                seg[0] = k
                seg[1] = j
                segments.append(seg)
                k=j

            else:
                j+=1
        except:
            j+=1
            
            t_visit.append(time[k])
            seg[0] = k
            seg[1] = j
            segments.append(seg)
            k=j
            
    return t_visit, segments