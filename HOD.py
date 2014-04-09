#Duncan Campbell
#April 7, 2014
#Calculate the halo occupation function

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
