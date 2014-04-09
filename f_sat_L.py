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
    
    centrals   = np.where(mock['ID_host']==-1)[0]
    satellites = np.where(mock['ID_host']!=-1)[0]

    #galaxy color
    color = mock['g-r']
    LHS   = 0.21-0.03*mock['M_r,0.1']
    blue  = np.where(color<LHS)[0] #indices of blue galaxies
    red   = np.where(color>LHS)[0] #indicies of red galaxies
    
    S_r  = 4.64 #solar constant
    Lgal = solar_lum(mock['M_r,0.1'],S_r)

    mask = np.zeros(len(mock))
    mask[red]=1
    f_sat_red = f_sat_L(Lgal,centrals,satellites,bins,mask) 
    mask = np.zeros(len(mock))
    mask[blue]=1
    f_sat_blue = f_sat_L(Lgal,centrals,satellites,bins,mask)

    plt.figure()
    plt.plot(bin_centers,f_sat_red,color='red')
    plt.plot(bin_centers,f_sat_blue,color='blue')
    plt.ylim([0,1])
    plt.show()

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
