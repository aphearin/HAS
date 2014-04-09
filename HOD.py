#Duncan Campbell
#April 7, 2014
#Calculate the halo transiton probability function.

def main():
    import numpy as np
    import h5py
    import matplotlib.pyplot as plt
    import custom_utilities as cu
    import sys

    catalogue = sys.argv[1]

    bins = np.arange(10,15,0.1)
    bin_centers = (bins[:-1]+bins[1:])/2.0

    print 'opening mock catalogue:', catalogue+'.hdf5'
    #open catalogue
    filepath_mock = cu.get_output_path() + 'processed_data/hearin_mocks/custom_catalogues/'
    f1 = h5py.File(filepath_mock+catalogue+'_extended.hdf5', 'r') #open catalogue file
    mock = f1.get(catalogue+'_extended')
    mock = np.array(mock)
    print 'length:', len(mock)
    for name in mock.dtype.names: print '     ', name
    
    occupied = np.where(mock['M_r,0.1']!=-99)
    empty = np.where(mock['M_r,0.1']==-99)
    host =  np.where(mock['ID_host']==-1)[0]
    color = mock['g-r']
    LHS = 0.7 - 0.032*(mock['M_r,0.1']+16.5) #Weinmann 2006
    blue  = np.where((color<LHS) & (mock['M_r,0.1']!=-99))[0]
    red   = np.where((color>LHS) & (mock['M_r,0.1']!=-99))[0]

    print len(red), len(blue)

    Ngal = np.zeros(len(mock))
    Ngal[occupied]=1
    host_ID = mock['ID_host']
    host_ID[host]=mock['ID_halo'][host]
    
    mask = np.zeros(len(mock))
    mask[:] = 1
    avg_N = hod(host_ID,mock['M_host'],Ngal,bins,mask)
    print sum(Ngal[np.where(mask==0)[0]])

    mask = np.zeros(len(mock))
    mask[red] = 1
    avg_N_red = hod(host_ID,mock['M_host'],Ngal,bins,mask)
    print sum(Ngal[np.where(mask==0)[0]])

    mask = np.zeros(len(mock))
    mask[blue] = 1
    avg_N_blue = hod(host_ID,mock['M_host'],Ngal,bins,mask)
    print sum(Ngal[np.where(mask==0)[0]])

    plt.figure()
    plt.plot(bin_centers,avg_N, color='black')
    plt.plot(bin_centers,avg_N_red, color='red')
    plt.plot(bin_centers,avg_N_blue, color='blue')
    plt.yscale('log')
    plt.ylim([0.01,500])
    plt.ylabel()
    plt.show()

def hod(halo_ID, halo_mass, Ngal, bins, mask):
    #Returns the average number of galaxies per host.
    #halo_ID: integer identification number of galaxy host ID
    #halo_mass: mass of galaxy host
    #Ngal: number of galaxies per entry. 0 if empty, [1,N] if occupied
    #bins: halo mass bins
    #mask: calculate for subset of galaxies
    import numpy as np
    Ngal = Ngal.copy()

    avg_N  = np.zeros((len(bins)-1)) #store result

    Ngal[np.where(mask==0)[0]]=0.0

    IDs, inds = np.unique(halo_ID, return_index=True)
    N = np.bincount(halo_ID, weights=Ngal)
    N = N[halo_ID]

    M = halo_mass[inds] #halo masses
    N = N[inds] #number of galaxies per halo

    result = np.digitize(M,bins)
    for i in range(0,len(bins)-1):
        inds = np.where(result==i+1)[0]
        N_halo = float(len(inds))
        N_gal = float(sum(N[inds]))
        if N_halo==0: avg_N[i]=0.0
        else: avg_N[i] = N_gal/N_halo

    return avg_N
    

if __name__ == '__main__':
    main()
