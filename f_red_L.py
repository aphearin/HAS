#!/usr/bin/env python

#import modules
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
import custom_utilities as cu
from astropy import cosmology
import sys

def main():
    group_cat = 'tinker'
    catalogue = sys.argv[1]

    bins = np.arange(9.5,10.8,0.1)
    bin_centers = (bins[:-1]+bins[1:])/2.0

    filepath_mock = cu.get_output_path() + 'processed_data/hearin_mocks/custom_catalogues/'
    print 'opening mock catalogue:', catalogue+'.hdf5'
    #open catalogue
    f1 = h5py.File(filepath_mock+catalogue+'.hdf5', 'r') #open catalogue file
    mock = f1.get(catalogue)
    
    centrals   = np.where(mock['ID_host']==-1)
    satellites = np.where(mock['ID_host']!=-1)

    #galaxy color
    color = mock['g-r']
    LHS   = 0.21-0.03*mock['M_r,0.1']
    blue  = np.where(color<LHS)[0] #indices of blue galaxies
    red   = np.where(color>LHS)[0] #indicies of red galaxies
    
    S_r  = 4.64 #solar constant
    Lgal = solar_lum(mock['M_r,0.1'],S_r)

    mask = np.zeros(len(mock))
    mask[centrals]=1
    f_red_cen = f_red_L(Lgal,red,blue,bins,mask) 
    mask = np.zeros(len(mock))
    mask[satellites]=1
    f_red_sat = f_red_L(Lgal,red,blue,bins,mask)

    print f_red_cen
    print f_red_sat

    plt.figure()
    plt.plot(bin_centers,f_red_cen,color='green')
    plt.plot(bin_centers,f_red_sat,color='yellow')
    plt.ylim([0,1])
    plt.show()

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

if __name__=='__main__':
    main()
