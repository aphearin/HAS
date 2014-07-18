#!/usr/bin/env python

from __future__ import division
import numpy as np
import custom_utilities as cu
from astropy.io import ascii
from astropy.table import Table


def main():
    
    sample1 = cu.sample_ra_dec_box(30,39, -12,-4, 5000)
    sample2 = cu.sample_ra_dec_box(30,39, -12,-4, 100000)
    
    ra1,dec1 = zip(*sample1)
    ra1 = np.array(ra1)
    dec1 = np.array(dec1)
    z1 = np.random.random(len(ra1))*0.2
    ra1 = ra1[z1>0.02]
    dec1 = dec1[z1>0.02]
    z1 = z1[z1>0.02]
    
    ra2,dec2 = zip(*sample2)
    ra2 = np.array(ra2)
    dec2 = np.array(dec2)
    
    ran_sample2 = cu.sample_ra_dec_box(30,39, -12,-4, 1000000)
    
    ran_ra2,ran_dec2 = zip(*ran_sample2)
    ran_ra2 = np.array(ran_ra2)
    ran_dec2 = np.array(ran_dec2)
    
    data = Table([ra1, dec1, z1], names=['ra', 'dec', 'z'])
    ascii.write(data, 'test_radec_1.dat')
    
    data = Table([ra2, dec2], names=['ra', 'dec'])
    ascii.write(data, 'test_radec_2.dat')
    
    data = Table([ran_ra2, ran_dec2], names=['ra', 'dec'])
    ascii.write(data, 'test_ran_radec_2.dat')
    
    
    

if __name__ == '__main__':
    main()