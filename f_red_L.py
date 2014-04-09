#Duncan Campbell
#Yale University
#April 8, 2014
#calculate the red fraction of galaxies as a function of galaxy luminosity

def f_red_L(Lgal,red,blue,bins,mask):
    #returns the red fraction of galaxies in luminosity bins
    import numpy as np
    f_red = np.zeros(len(bins)-1)

    red_mask = mask[red]
    blue_mask = mask[blue]
    red = red[np.where(red_mask==1)[0]].copy()
    blue = blue[np.where(blue_mask==1)[0]].copy()

    result = np.digitize(Lgal,bins=bins)
    for i in range(0,len(bins)-1):
        ind = np.where(result==i+1)[0]
        red_gal  = ind[np.in1d(ind,red)]
        blue_gal = ind[np.in1d(ind,blue)]
        print len(red_gal), len(blue_gal)
        if len(ind)>0:
            f_red[i] = float(len(red_gal))/(len(red_gal)+len(blue_gal))
        else: f_red[i]=0.0

    return f_red

def solar_lum(M,Msol):
    L = ((Msol-M)/2.5)
    return L  
