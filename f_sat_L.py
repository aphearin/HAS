#Duncan Campbell
#Yale University
#April 8, 2014
#calculate the satellite fraction fo galaxies in a list as a function of galaxy luminosity

def f_sat_L(Lgal,cen,sat,bins,mask):
    #returns the satellite fraction of galaxies in luminosity bins
    import numpy as np
    f_sat = np.zeros(len(bins)-1)

    sat_mask = mask[sat]
    cen_mask = mask[cen]
    sat = sat[np.where(sat_mask==1)[0]].copy()
    cen = cen[np.where(cen_mask==1)[0]].copy()

    result = np.digitize(Lgal,bins=bins)
    for i in range(0,len(bins)-1):
        ind = np.where(result==i+1)[0]
        sat_gal  = ind[np.in1d(ind,sat)]
        cen_gal = ind[np.in1d(ind,cen)]
        if len(ind)>0:
            f_sat[i] = float(len(sat_gal))/(len(cen_gal)+len(sat_gal))
        else: f_sat[i]=0.0

    return f_sat

def solar_lum(M,Msol):
    L = ((Msol-M)/2.5)
    return L  

if __name__=='__main__':
    main()
