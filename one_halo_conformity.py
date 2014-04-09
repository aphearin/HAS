 #!/usr/bin/env python

import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
import custom_utilities as cu
from astropy import cosmology
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable

def main():
    catalogue_1 = sys.argv[1]
    catalogue_2 = sys.argv[2]

    N_boots = 50

    #set up ploting parameters
    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(6.95, 0.9*5.61))
    fig.subplots_adjust(hspace=0, wspace=0.05)
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
    axes = axes.flatten()

    #plot 1
    axes[0].set_xlim([12,15])
    axes[0].set_ylim([0,1])
    axes[0].set_ylabel('${f}_{blue}$')
    axes[0].set_xticks([12.5,13,13.5,14,14.5])
    axes[0].set_yticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    axes[0].set_title('Berlind FoF groups')
    axes[0].text(12.2,0.85,'age matching mock', fontsize=10)
    #plot 2
    axes[1].set_xlim([12,15])
    axes[1].set_ylim([0,1])
    axes[1].set_title('Tinker SO groups')
    #plot 3
    axes[2].set_xlim([12,15])
    axes[2].set_ylim([0,1])
    axes[2].set_title('Yang SO groups')
    #plot 4
    axes[3].set_xlim([12,15])
    axes[3].set_ylim([0,1])
    axes[3].set_xlabel('$\log(M)$ $[{M}_{\odot}/h]$')
    axes[3].set_ylabel('${f}_{blue}$')
    axes[3].text(12.2,0.85,'shuffled mock', fontsize=10)
    #plot 5
    axes[4].set_xlim([12,15])
    axes[4].set_ylim([0,1])
    axes[4].set_xlabel('$\log(M)$ $[{M}_{\odot}/h]$')
    #plot 6
    axes[5].set_xlim([12,15])
    axes[5].set_ylim([0,1])
    axes[5].set_xlabel('$\log(M)$ $[{M}_{\odot}/h]$')
    
    catalogue = catalogue_1

    print 'opening mock catalogue:', catalogue+'.hdf5'
    #open catalogue
    filepath_cat = cu.get_output_path() + 'processed_data/hearin_mocks/custom_catalogues/'
    f1 = h5py.File(filepath_cat+catalogue+'.hdf5', 'r') #open catalogue file
    GC = f1.get(catalogue)

    #galaxy color
    h=1.0
    color = GC['g-r']
    #LHS = 0.7 - 0.032*(GC['M_r,0.1']-5.0*np.log10(h)+16.5) #Weinmann 2006
    LHS = 0.21-0.03*GC['M_r,0.1']
    blue = np.where(color<LHS)[0] #indices of blue galaxies
    red = np.where(color>LHS)[0] #indicies of red galaxies
    lookup = color<LHS #true if blue, false if red
    flip = (lookup == False)

    centrals = np.where(GC['ID_host'] == -1)[0] #central galaxies
    N_groups = len(centrals)
    
    mass_bins = np.arange(10,15.2,0.2) #log mass bins
    mass_hist, bins = np.histogram(GC['M_host'][centrals], bins=mass_bins) #group histogram by log(mass)
    mass_bin_ind = np.digitize(GC['M_host'][centrals], bins=mass_bins) #indices of groups in log(mass) bins
    selection = np.where(lookup==True)[0]
    selection = np.intersect1d(selection,centrals)
    mass_hist_blue, bins = np.histogram(GC['M_host'][selection], bins=mass_bins) #group histogram by log(mass)
    mass_bin_ind_blue = np.digitize(GC['M_host'][selection], bins=mass_bins) #indices of groups in log(mass) bins
    selection = np.where(lookup==False)[0]
    selection = np.intersect1d(selection,centrals)
    mass_hist_red, bins = np.histogram(GC['M_host'][selection], bins=mass_bins) #group histogram by log(mass)
    mass_bin_ind_red = np.digitize(GC['M_host'][selection], bins=mass_bins) #indices of groups in log(mass) bins
    width = 1.0 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    #normalize the hists
    mass_hist_red = mass_hist_red/float(N_groups)
    mass_hist_blue = mass_hist_blue/float(N_groups)

    f_cen_blue   = np.zeros(len(mass_bins)-1) #fraction of blue satellites for blue centrals
    f_cen_red    = np.zeros(len(mass_bins)-1) #fraction of blue satellites for red centrals
    err_cen_blue = np.zeros(len(mass_bins)-1) #poisson error on blue fraction for blue centrals
    err_cen_red  = np.zeros(len(mass_bins)-1) #poisson error on blue fraction for red centrals
    N_cen_blue   = np.zeros(len(mass_bins)-1) #number of satellites for blue centrals
    N_cen_red    = np.zeros(len(mass_bins)-1) #number of satellites for red centrals
    N_groups_blue= np.zeros(len(mass_bins)-1) #number of satellites for red centrals
    N_groups_red = np.zeros(len(mass_bins)-1) #number of satellites for red centrals
    for i in range(0,len(mass_bins)-1):
        print i, mass_bins[i], mass_bins[i+1]
        ind = np.where(mass_bin_ind==i+1)[0]
        if len(ind)>0:
            print 'number of groups:', len(ind)
            ids = GC['ID_halo'][centrals[ind]]
            sat_galaxy_members = np.in1d(GC['ID_host'],ids) #satellite galaxies in the mass bin
            sat_galaxy_members = np.where(sat_galaxy_members)[0] #indicies of galaxies
            cen_galaxy_members = np.in1d(GC['ID_halo'],ids) #central galaxies in the mass bin
            cen_galaxy_members = np.where(cen_galaxy_members)[0] #indicies of galaxies
            galaxy_members = np.hstack((sat_galaxy_members,cen_galaxy_members))
            print 'number of galaxies:', len(galaxy_members)
            satellite_members = np.where(GC['ID_host'][galaxy_members]!=-1)[0] #satellites
            satellite_members = galaxy_members[satellite_members] #indices of satellites
            central_members = np.where(GC['ID_host'][galaxy_members]==-1)[0] #centrals
            central_members = galaxy_members[central_members] #indices of centrals
            print 'number of centrals:', len(central_members)
            print 'number of satellites:', len(satellite_members)
            print 'check:',  len(central_members) + len(satellite_members) == len(galaxy_members)
            blue_central_members = np.where(lookup[central_members]==True)[0] #blue centrals
            blue_central_members = central_members[blue_central_members] #indicies of blue centrals
            red_central_members = np.where(lookup[central_members]==False)[0] #red centrals
            red_central_members = central_members[red_central_members] #indicies of red centrals

            print 'number of blue centrals:', len(blue_central_members)
            print 'number of red centrals:', len(red_central_members) 
            print 'check:', len(blue_central_members)+len(red_central_members) == len(central_members)  
            
            blue_central_satellites = np.in1d(GC['ID_host'][satellite_members],GC['ID_halo'][blue_central_members])
            blue_central_satellites = np.where(blue_central_satellites)[0]
            blue_central_satellites = satellite_members[blue_central_satellites]
            red_central_satellites = np.in1d(GC['ID_host'][satellite_members],GC['ID_halo'][red_central_members])
            red_central_satellites = np.where(red_central_satellites)[0]
            red_central_satellites = satellite_members[red_central_satellites]
            
            print 'number of blue central satellites:', len(blue_central_satellites)
            print 'number of red central satellites:', len(red_central_satellites)
            print 'check:', len(blue_central_satellites) + len(red_central_satellites) == len(satellite_members)

            blue_central_blue_satellites = np.where(lookup[blue_central_satellites]==True)[0]
            blue_central_blue_satellites = blue_central_satellites[blue_central_blue_satellites]
            blue_central_red_satellites = np.where(lookup[blue_central_satellites]==False)[0]
            blue_central_red_satellites = blue_central_satellites[blue_central_red_satellites]
            red_central_blue_satellites = np.where(lookup[red_central_satellites]==True)[0]
            red_central_blue_satellites = red_central_satellites[red_central_blue_satellites]
            red_central_red_satellites = np.where(lookup[red_central_satellites]==False)[0]
            red_central_red_satellites = red_central_satellites[red_central_red_satellites]

            N_blue_central_blue_satellites  = float(len(blue_central_blue_satellites))
            N_blue_central_red_satellites = float(len(blue_central_red_satellites))
            N_red_central_blue_satellites  = float(len(red_central_blue_satellites))
            N_red_central_red_satellites = float(len(red_central_red_satellites))
            
            print 'check:', N_blue_central_blue_satellites+N_blue_central_red_satellites==len(blue_central_satellites)
            print 'check:', N_red_central_blue_satellites+N_red_central_red_satellites==len(red_central_satellites)

            if len(blue_central_satellites)>0: 
                f_cen_blue[i] = N_blue_central_blue_satellites / \
                             (N_blue_central_blue_satellites + N_blue_central_red_satellites)
                print 'f_cen_blue:', f_cen_blue[i]
            if len(red_central_satellites)>0:
                f_cen_red[i] = N_red_central_blue_satellites / \
                            (N_red_central_blue_satellites + N_red_central_red_satellites)
                print 'f_cen_red:', f_cen_red[i]

            #number satellites in the bin
            N_cen_blue[i] = float(len(blue_central_satellites))
            N_cen_red[i] = float(len(red_central_satellites))
            #number of centrals in the bin
            N_groups_blue[i] = float(len(blue_central_members))
            N_groups_red[i] = float(len(red_central_members))
   

    ax=axes[0]
    selection = np.where( N_cen_blue > 50)[0]
    p3, = ax.plot(mass_bins[0:-1][selection],f_cen_blue[selection], color='blue', alpha=1)
    selection = np.where( N_cen_red > 50)[0]
    p4, = ax.plot(mass_bins[0:-1][selection],f_cen_red[selection], color='red', alpha=1)
    ax=axes[1]
    selection = np.where( N_cen_blue > 50)[0]
    p3, = ax.plot(mass_bins[0:-1][selection],f_cen_blue[selection], color='blue', alpha=1)
    selection = np.where( N_cen_red > 50)[0]
    p4, = ax.plot(mass_bins[0:-1][selection],f_cen_red[selection], color='red', alpha=1)
    ax=axes[2]
    selection = np.where( N_cen_blue > 50)[0]
    p3, = ax.plot(mass_bins[0:-1][selection],f_cen_blue[selection], color='blue', alpha=1)
    selection = np.where( N_cen_red > 50)[0]
    p4, = ax.plot(mass_bins[0:-1][selection],f_cen_red[selection], color='red', alpha=1)
    
    #################################################################################################

    group_cat = 'berlind_mocks'

    if group_cat == 'tinker_mocks':
        filepath_cat = cu.get_output_path() + 'processed_data/tinker_groupcat/mock_runs/4th_run/custom_catalogues/'
        savepath = cu.get_output_path() + 'analysis/tinker_groupcat/'
        plotpath = cu.get_plot_path() + 'analysis/tinker_groupcat/'
        group_catalogue = catalogue+'_clf_groups_M19'
    if group_cat == 'berlind_mocks':
        filepath_cat = cu.get_output_path() + 'processed_data/berlind_groupcat/mock_runs/4th_run/custom_catalogues/'
        savepath = cu.get_output_path() + 'analysis/berlind_groupcat/'
        plotpath = cu.get_plot_path() + 'analysis/berlind_groupcat/'
        group_catalogue = catalogue+'_groups'
    if group_cat == 'yang_mocks':
        filepath_cat = cu.get_output_path() + 'processed_data/yang_groupcat/mock_runs/4th_run/custom_catalogues/'
        savepath = cu.get_output_path() + 'analysis/yang_groupcat/'
        plotpath = cu.get_plot_path() + 'analysis/yang_groupcat/'
        group_catalogue = catalogue+'_groups'

    mass_bins = np.arange(10,15.2,0.2) #log mass bins
    f_cen_blue   = np.zeros((N_boots,len(mass_bins)-1)) #fraction of blue satellites for blue centrals
    f_cen_red    = np.zeros((N_boots,len(mass_bins)-1)) #fraction of blue satellites for red centrals
    N_cen_blue   = np.zeros((N_boots,len(mass_bins)-1)) #number of satellites for blue centrals
    N_cen_red    = np.zeros((N_boots,len(mass_bins)-1)) #number of satellites for red centrals
    N_groups_blue= np.zeros((N_boots,len(mass_bins)-1)) #number of satellites for red centrals
    N_groups_red = np.zeros((N_boots,len(mass_bins)-1)) #number of satellites for red centrals

    for boot in range(0,N_boots):

        print 'opening group catalogue:', catalogue
        #open catalogue
        filepath = filepath_cat+'bootstraps/'
        catalogue_1 = group_catalogue+'_'+str(boot)
        print catalogue_1
        f =  h5py.File(filepath+catalogue_1+'.hdf5', 'r') #open catalogue file
        GC = f.get(catalogue_1)
        print len(GC)

        #do we have a measure of M_u?
        good = np.where(GC['M_g,0.1']!=-99.9)[0]
        bad  = np.where(GC['M_g,0.1']==-99.9)[0]
        print 'number of galaxies without M_u:', len(bad)    

        #galaxy color
        h=1.0
        color = GC['M_g,0.1'] - GC['M_r,0.1']
        #LHS = 0.7 - 0.032*(GC['M_r,0.1']-5.0*np.log10(h)+16.5) #Weinmann 2006
        LHS = 0.21-0.03*GC['M_r,0.1']
        blue = np.where(color<LHS)[0] #indices of blue galaxies
        red = np.where(color>LHS)[0] #indicies of red galaxies
        lookup = color<LHS #true if blue, false if red
        flip = (lookup == False)

        print 'number of blue galaxies:', len(blue)
        print 'number of red galaxies:', len(red)

        centrals = np.where(GC['RPROJ'][good] == 0)[0] #central galaxies
        centrals = good[centrals]
        bad_centrals = np.where(GC['RPROJ'][bad] == 0)[0] #central galaxies with ssfr measured
        bad_centrals = bad[bad_centrals]
        N_groups = len(centrals)+len(bad_centrals)
    
        mass_hist, bins = np.histogram(GC['MGROUP'][centrals], bins=mass_bins) #group histogram by log(mass)
        mass_bin_ind = np.digitize(GC['MGROUP'][centrals], bins=mass_bins) #indices of groups in log(mass) bins
        selection = np.where(lookup==True)[0]
        selection = np.intersect1d(selection,centrals)
        mass_hist_blue, bins = np.histogram(GC['MGROUP'][selection], bins=mass_bins) #group histogram by log(mass)
        mass_bin_ind_blue = np.digitize(GC['MGROUP'][selection], bins=mass_bins) #indices of groups in log(mass) bins
        selection = np.where(lookup==False)[0]
        selection = np.intersect1d(selection,centrals)
        mass_hist_red, bins = np.histogram(GC['MGROUP'][selection], bins=mass_bins) #group histogram by log(mass)
        mass_bin_ind_red = np.digitize(GC['MGROUP'][selection], bins=mass_bins) #indices of groups in log(mass) bins
        selection = bad_centrals
        mass_hist_grey, bins = np.histogram(GC['MGROUP'][selection], bins=mass_bins) #group histogram by log(mass)
        if len(bad_centrals)>0:
            mass_bin_ind_grey = np.digitize(GC['MGROUP'][selection], bins=mass_bins) #indices of groups in log(mass) bins
        width = 1.0 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        #normalize the hists
        mass_hist_grey = mass_hist_grey/float(N_groups)
        mass_hist_red = mass_hist_red/float(N_groups)
        mass_hist_blue = mass_hist_blue/float(N_groups)

    
        for i in range(0,len(mass_bins)-1):
            print i, mass_bins[i], mass_bins[i+1]
            ind = np.where(mass_bin_ind==i+1)[0]
            if len(ind)>0:
                print 'number of groups:', len(ind)
                ids = GC['GROUP_ID'][centrals[ind]]
                galaxy_members = np.in1d(GC['GROUP_ID'],ids) #galaxies in the mass bin
                galaxy_members = np.where(galaxy_members)[0] #indicies of galaxies
                print 'number of galaxies:', len(galaxy_members)
                satellite_members = np.where(GC['RPROJ'][galaxy_members]>0)[0] #satellites
                satellite_members = galaxy_members[satellite_members] #indices of satellites
                central_members = np.where(GC['RPROJ'][galaxy_members]==0)[0] #centrals
                central_members = galaxy_members[central_members] #indices of centrals
                print 'number of centrals:', len(central_members)
                print 'number of satellites:', len(satellite_members)
                print 'check:',  len(central_members) + len(satellite_members) == len(galaxy_members)
                blue_central_members = np.where(lookup[central_members]==True)[0] #blue centrals
                blue_central_members = central_members[blue_central_members] #indicies of blue centrals
                red_central_members = np.where(lookup[central_members]==False)[0] #red centrals
                red_central_members = central_members[red_central_members] #indicies of red centrals

                print 'number of blue centrals:', len(blue_central_members)
                print 'number of red centrals:', len(red_central_members) 
                print 'check:', len(blue_central_members)+len(red_central_members) == len(central_members)  
            
                blue_central_satellites = np.in1d(GC['GROUP_ID'][satellite_members],GC['GROUP_ID'][blue_central_members])
                blue_central_satellites = np.where(blue_central_satellites)[0]
                blue_central_satellites = satellite_members[blue_central_satellites]
                red_central_satellites = np.in1d(GC['GROUP_ID'][satellite_members],GC['GROUP_ID'][red_central_members])
                red_central_satellites = np.where(red_central_satellites)[0]
                red_central_satellites = satellite_members[red_central_satellites]
            
                print 'number of blue central satellites:', len(blue_central_satellites)
                print 'number of red central satellites:', len(red_central_satellites)
                print 'check:', len(blue_central_satellites) + len(red_central_satellites) == len(satellite_members)

                blue_central_blue_satellites = np.where(lookup[blue_central_satellites]==True)[0]
                blue_central_blue_satellites = blue_central_satellites[blue_central_blue_satellites]
                blue_central_red_satellites = np.where(lookup[blue_central_satellites]==False)[0]
                blue_central_red_satellites = blue_central_satellites[blue_central_red_satellites]
                red_central_blue_satellites = np.where(lookup[red_central_satellites]==True)[0]
                red_central_blue_satellites = red_central_satellites[red_central_blue_satellites]
                red_central_red_satellites = np.where(lookup[red_central_satellites]==False)[0]
                red_central_red_satellites = red_central_satellites[red_central_red_satellites]

                N_blue_central_blue_satellites  = float(len(blue_central_blue_satellites))
                N_blue_central_red_satellites = float(len(blue_central_red_satellites))
                N_red_central_blue_satellites  = float(len(red_central_blue_satellites))
                N_red_central_red_satellites = float(len(red_central_red_satellites))
            
                print 'check:', N_blue_central_blue_satellites+N_blue_central_red_satellites==len(blue_central_satellites)
                print 'check:', N_red_central_blue_satellites+N_red_central_red_satellites==len(red_central_satellites)

                if len(blue_central_satellites)>0: 
                    f_cen_blue[boot,i] = N_blue_central_blue_satellites / \
                             (N_blue_central_blue_satellites + N_blue_central_red_satellites)
                    print 'f_cen_blue:', f_cen_blue[boot,i]
                if len(red_central_satellites)>0:
                    f_cen_red[boot,i] = N_red_central_blue_satellites / \
                            (N_red_central_blue_satellites + N_red_central_red_satellites)
                    print 'f_cen_red:', f_cen_red[boot,i]

                #number satellites in the bin
                N_cen_blue[boot,i] = float(len(blue_central_satellites))
                N_cen_red[boot,i] = float(len(red_central_satellites))
                #number of centrals in the bin
                N_groups_blue[boot,i] = float(len(blue_central_members))
                N_groups_red[boot,i] = float(len(red_central_members))

    f_blue = np.nanmean(f_cen_blue, axis=0)
    f_red = np.nanmean(f_cen_red, axis=0)
    N_blue = np.nanmean(N_cen_blue, axis=0)
    N_red = np.nanmean(N_cen_red, axis=0)
    err_blue = np.nanstd(f_cen_blue, axis=0)
    err_red = np.nanstd(f_cen_red, axis=0)
    
    ax=axes[0]
    selection = np.where( N_blue > 50)[0]
    p1=ax.errorbar(mass_bins[0:-1][selection],f_blue[selection],yerr=err_blue[selection], fmt='o', color='blue', mec='none', ms=3)
    selection = np.where( N_red > 50)[0]
    p2=ax.errorbar(mass_bins[0:-1][selection],f_red[selection], yerr=err_red[selection], fmt='o', color='red', mec='none', ms=3)


    ######################################################################################

    group_cat = 'tinker_mocks'

    if group_cat == 'tinker_mocks':
        filepath_cat = cu.get_output_path() + 'processed_data/tinker_groupcat/mock_runs/4th_run/custom_catalogues/'
        savepath = cu.get_output_path() + 'analysis/tinker_groupcat/'
        plotpath = cu.get_plot_path() + 'analysis/tinker_groupcat/'
        group_catalogue = catalogue+'_clf_groups_M19'
    if group_cat == 'berlind_mocks':
        filepath_cat = cu.get_output_path() + 'processed_data/berlind_groupcat/mock_runs/4th_run/custom_catalogues/'
        savepath = cu.get_output_path() + 'analysis/berlind_groupcat/'
        plotpath = cu.get_plot_path() + 'analysis/berlind_groupcat/'
        group_catalogue = catalogue+'_groups'
    if group_cat == 'yang_mocks':
        filepath_cat = cu.get_output_path() + 'processed_data/yang_groupcat/mock_runs/4th_run/custom_catalogues/'
        savepath = cu.get_output_path() + 'analysis/yang_groupcat/'
        plotpath = cu.get_plot_path() + 'analysis/yang_groupcat/'
        group_catalogue = catalogue+'_groups'

    mass_bins = np.arange(10,15.2,0.2) #log mass bins
    f_cen_blue   = np.zeros((N_boots,len(mass_bins)-1)) #fraction of blue satellites for blue centrals
    f_cen_red    = np.zeros((N_boots,len(mass_bins)-1)) #fraction of blue satellites for red centrals
    N_cen_blue   = np.zeros((N_boots,len(mass_bins)-1)) #number of satellites for blue centrals
    N_cen_red    = np.zeros((N_boots,len(mass_bins)-1)) #number of satellites for red centrals
    N_groups_blue= np.zeros((N_boots,len(mass_bins)-1)) #number of satellites for red centrals
    N_groups_red = np.zeros((N_boots,len(mass_bins)-1)) #number of satellites for red centrals

    for boot in range(0,N_boots):

        print 'opening group catalogue:', catalogue
        #open catalogue
        filepath = filepath_cat+'bootstraps/'
        catalogue_1 = group_catalogue+'_'+str(boot)
        print catalogue_1
        f =  h5py.File(filepath+catalogue_1+'.hdf5', 'r') #open catalogue file
        GC = f.get(catalogue_1)
        print len(GC)

        #do we have a measure of M_u?
        good = np.where(GC['M_g,0.1']!=-99.9)[0]
        bad  = np.where(GC['M_g,0.1']==-99.9)[0]
        print 'number of galaxies without M_u:', len(bad)    

        #galaxy color
        h=1.0
        color = GC['M_g,0.1'] - GC['M_r,0.1']
        #LHS = 0.7 - 0.032*(GC['M_r,0.1']-5.0*np.log10(h)+16.5) #Weinmann 2006
        LHS = 0.21-0.03*GC['M_r,0.1']
        blue = np.where(color<LHS)[0] #indices of blue galaxies
        red = np.where(color>LHS)[0] #indicies of red galaxies
        lookup = color<LHS #true if blue, false if red
        flip = (lookup == False)

        print 'number of blue galaxies:', len(blue)
        print 'number of red galaxies:', len(red)

        centrals = np.where(GC['RPROJ'][good] == 0)[0] #central galaxies
        centrals = good[centrals]
        bad_centrals = np.where(GC['RPROJ'][bad] == 0)[0] #central galaxies with ssfr measured
        bad_centrals = bad[bad_centrals]
        N_groups = len(centrals)+len(bad_centrals)
    
        mass_hist, bins = np.histogram(GC['MGROUP'][centrals], bins=mass_bins) #group histogram by log(mass)
        mass_bin_ind = np.digitize(GC['MGROUP'][centrals], bins=mass_bins) #indices of groups in log(mass) bins
        selection = np.where(lookup==True)[0]
        selection = np.intersect1d(selection,centrals)
        mass_hist_blue, bins = np.histogram(GC['MGROUP'][selection], bins=mass_bins) #group histogram by log(mass)
        mass_bin_ind_blue = np.digitize(GC['MGROUP'][selection], bins=mass_bins) #indices of groups in log(mass) bins
        selection = np.where(lookup==False)[0]
        selection = np.intersect1d(selection,centrals)
        mass_hist_red, bins = np.histogram(GC['MGROUP'][selection], bins=mass_bins) #group histogram by log(mass)
        mass_bin_ind_red = np.digitize(GC['MGROUP'][selection], bins=mass_bins) #indices of groups in log(mass) bins
        selection = bad_centrals
        mass_hist_grey, bins = np.histogram(GC['MGROUP'][selection], bins=mass_bins) #group histogram by log(mass)
        if len(bad_centrals)>0:
            mass_bin_ind_grey = np.digitize(GC['MGROUP'][selection], bins=mass_bins) #indices of groups in log(mass) bins
        width = 1.0 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        #normalize the hists
        mass_hist_grey = mass_hist_grey/float(N_groups)
        mass_hist_red = mass_hist_red/float(N_groups)
        mass_hist_blue = mass_hist_blue/float(N_groups)

    
        for i in range(0,len(mass_bins)-1):
            print i, mass_bins[i], mass_bins[i+1]
            ind = np.where(mass_bin_ind==i+1)[0]
            if len(ind)>0:
                print 'number of groups:', len(ind)
                ids = GC['GROUP_ID'][centrals[ind]]
                galaxy_members = np.in1d(GC['GROUP_ID'],ids) #galaxies in the mass bin
                galaxy_members = np.where(galaxy_members)[0] #indicies of galaxies
                print 'number of galaxies:', len(galaxy_members)
                satellite_members = np.where(GC['RPROJ'][galaxy_members]>0)[0] #satellites
                satellite_members = galaxy_members[satellite_members] #indices of satellites
                central_members = np.where(GC['RPROJ'][galaxy_members]==0)[0] #centrals
                central_members = galaxy_members[central_members] #indices of centrals
                print 'number of centrals:', len(central_members)
                print 'number of satellites:', len(satellite_members)
                print 'check:',  len(central_members) + len(satellite_members) == len(galaxy_members)
                blue_central_members = np.where(lookup[central_members]==True)[0] #blue centrals
                blue_central_members = central_members[blue_central_members] #indicies of blue centrals
                red_central_members = np.where(lookup[central_members]==False)[0] #red centrals
                red_central_members = central_members[red_central_members] #indicies of red centrals

                print 'number of blue centrals:', len(blue_central_members)
                print 'number of red centrals:', len(red_central_members) 
                print 'check:', len(blue_central_members)+len(red_central_members) == len(central_members)  
            
                blue_central_satellites = np.in1d(GC['GROUP_ID'][satellite_members],GC['GROUP_ID'][blue_central_members])
                blue_central_satellites = np.where(blue_central_satellites)[0]
                blue_central_satellites = satellite_members[blue_central_satellites]
                red_central_satellites = np.in1d(GC['GROUP_ID'][satellite_members],GC['GROUP_ID'][red_central_members])
                red_central_satellites = np.where(red_central_satellites)[0]
                red_central_satellites = satellite_members[red_central_satellites]
            
                print 'number of blue central satellites:', len(blue_central_satellites)
                print 'number of red central satellites:', len(red_central_satellites)
                print 'check:', len(blue_central_satellites) + len(red_central_satellites) == len(satellite_members)

                blue_central_blue_satellites = np.where(lookup[blue_central_satellites]==True)[0]
                blue_central_blue_satellites = blue_central_satellites[blue_central_blue_satellites]
                blue_central_red_satellites = np.where(lookup[blue_central_satellites]==False)[0]
                blue_central_red_satellites = blue_central_satellites[blue_central_red_satellites]
                red_central_blue_satellites = np.where(lookup[red_central_satellites]==True)[0]
                red_central_blue_satellites = red_central_satellites[red_central_blue_satellites]
                red_central_red_satellites = np.where(lookup[red_central_satellites]==False)[0]
                red_central_red_satellites = red_central_satellites[red_central_red_satellites]

                N_blue_central_blue_satellites  = float(len(blue_central_blue_satellites))
                N_blue_central_red_satellites = float(len(blue_central_red_satellites))
                N_red_central_blue_satellites  = float(len(red_central_blue_satellites))
                N_red_central_red_satellites = float(len(red_central_red_satellites))
            
                print 'check:', N_blue_central_blue_satellites+N_blue_central_red_satellites==len(blue_central_satellites)
                print 'check:', N_red_central_blue_satellites+N_red_central_red_satellites==len(red_central_satellites)

                if len(blue_central_satellites)>0: 
                    f_cen_blue[boot,i] = N_blue_central_blue_satellites / \
                             (N_blue_central_blue_satellites + N_blue_central_red_satellites)
                    print 'f_cen_blue:', f_cen_blue[boot,i]
                if len(red_central_satellites)>0:
                    f_cen_red[boot,i] = N_red_central_blue_satellites / \
                            (N_red_central_blue_satellites + N_red_central_red_satellites)
                    print 'f_cen_red:', f_cen_red[boot,i]

                #number satellites in the bin
                N_cen_blue[boot,i] = float(len(blue_central_satellites))
                N_cen_red[boot,i] = float(len(red_central_satellites))
                #number of centrals in the bin
                N_groups_blue[boot,i] = float(len(blue_central_members))
                N_groups_red[boot,i] = float(len(red_central_members))

    f_blue = np.nanmean(f_cen_blue, axis=0)
    f_red = np.nanmean(f_cen_red, axis=0)
    N_blue = np.nanmean(N_cen_blue, axis=0)
    N_red = np.nanmean(N_cen_red, axis=0)
    err_blue = np.nanstd(f_cen_blue, axis=0)
    err_red = np.nanstd(f_cen_red, axis=0)
    
    ax=axes[1]
    selection = np.where( N_blue > 50)[0]
    p1=ax.errorbar(mass_bins[0:-1][selection],f_blue[selection],yerr=err_blue[selection], fmt='o', color='blue', mec='none', ms=3)
    selection = np.where( N_red > 50)[0]
    p2=ax.errorbar(mass_bins[0:-1][selection],f_red[selection], yerr=err_red[selection], fmt='o', color='red', mec='none', ms=3)


    ######################################################################################


    group_cat = 'yang_mocks'

    if group_cat == 'tinker_mocks':
        filepath_cat = cu.get_output_path() + 'processed_data/tinker_groupcat/mock_runs/4th_run/custom_catalogues/'
        savepath = cu.get_output_path() + 'analysis/tinker_groupcat/'
        plotpath = cu.get_plot_path() + 'analysis/tinker_groupcat/'
        group_catalogue = catalogue+'_clf_groups_M19'
    if group_cat == 'berlind_mocks':
        filepath_cat = cu.get_output_path() + 'processed_data/berlind_groupcat/mock_runs/4th_run/custom_catalogues/'
        savepath = cu.get_output_path() + 'analysis/berlind_groupcat/'
        plotpath = cu.get_plot_path() + 'analysis/berlind_groupcat/'
        group_catalogue = catalogue+'_groups'
    if group_cat == 'yang_mocks':
        filepath_cat = cu.get_output_path() + 'processed_data/yang_groupcat/mock_runs/4th_run/custom_catalogues/'
        savepath = cu.get_output_path() + 'analysis/yang_groupcat/'
        plotpath = cu.get_plot_path() + 'analysis/yang_groupcat/'
        group_catalogue = catalogue+'_groups'

    mass_bins = np.arange(10,15.2,0.2) #log mass bins
    f_cen_blue   = np.zeros((N_boots,len(mass_bins)-1)) #fraction of blue satellites for blue centrals
    f_cen_red    = np.zeros((N_boots,len(mass_bins)-1)) #fraction of blue satellites for red centrals
    N_cen_blue   = np.zeros((N_boots,len(mass_bins)-1)) #number of satellites for blue centrals
    N_cen_red    = np.zeros((N_boots,len(mass_bins)-1)) #number of satellites for red centrals
    N_groups_blue= np.zeros((N_boots,len(mass_bins)-1)) #number of satellites for red centrals
    N_groups_red = np.zeros((N_boots,len(mass_bins)-1)) #number of satellites for red centrals

    for boot in range(0,N_boots):

        print 'opening group catalogue:', catalogue
        #open catalogue
        filepath = filepath_cat+'bootstraps/'
        catalogue_1 = group_catalogue+'_'+str(boot)
        print catalogue_1
        f =  h5py.File(filepath+catalogue_1+'.hdf5', 'r') #open catalogue file
        GC = f.get(catalogue_1)
        print len(GC)

        #do we have a measure of M_u?
        good = np.where(GC['M_g,0.1']!=-99.9)[0]
        bad  = np.where(GC['M_g,0.1']==-99.9)[0]
        print 'number of galaxies without M_u:', len(bad)    

        #galaxy color
        h=1.0
        color = GC['M_g,0.1'] - GC['M_r,0.1']
        #LHS = 0.7 - 0.032*(GC['M_r,0.1']-5.0*np.log10(h)+16.5) #Weinmann 2006
        LHS = 0.21-0.03*GC['M_r,0.1']
        blue = np.where(color<LHS)[0] #indices of blue galaxies
        red = np.where(color>LHS)[0] #indicies of red galaxies
        lookup = color<LHS #true if blue, false if red
        flip = (lookup == False)

        print 'number of blue galaxies:', len(blue)
        print 'number of red galaxies:', len(red)

        centrals = np.where(GC['RPROJ'][good] == 0)[0] #central galaxies
        centrals = good[centrals]
        bad_centrals = np.where(GC['RPROJ'][bad] == 0)[0] #central galaxies with ssfr measured
        bad_centrals = bad[bad_centrals]
        N_groups = len(centrals)+len(bad_centrals)
    
        mass_hist, bins = np.histogram(GC['MGROUP'][centrals], bins=mass_bins) #group histogram by log(mass)
        mass_bin_ind = np.digitize(GC['MGROUP'][centrals], bins=mass_bins) #indices of groups in log(mass) bins
        selection = np.where(lookup==True)[0]
        selection = np.intersect1d(selection,centrals)
        mass_hist_blue, bins = np.histogram(GC['MGROUP'][selection], bins=mass_bins) #group histogram by log(mass)
        mass_bin_ind_blue = np.digitize(GC['MGROUP'][selection], bins=mass_bins) #indices of groups in log(mass) bins
        selection = np.where(lookup==False)[0]
        selection = np.intersect1d(selection,centrals)
        mass_hist_red, bins = np.histogram(GC['MGROUP'][selection], bins=mass_bins) #group histogram by log(mass)
        mass_bin_ind_red = np.digitize(GC['MGROUP'][selection], bins=mass_bins) #indices of groups in log(mass) bins
        selection = bad_centrals
        mass_hist_grey, bins = np.histogram(GC['MGROUP'][selection], bins=mass_bins) #group histogram by log(mass)
        if len(bad_centrals)>0:
            mass_bin_ind_grey = np.digitize(GC['MGROUP'][selection], bins=mass_bins) #indices of groups in log(mass) bins
        width = 1.0 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        #normalize the hists
        mass_hist_grey = mass_hist_grey/float(N_groups)
        mass_hist_red = mass_hist_red/float(N_groups)
        mass_hist_blue = mass_hist_blue/float(N_groups)

    
        for i in range(0,len(mass_bins)-1):
            print i, mass_bins[i], mass_bins[i+1]
            ind = np.where(mass_bin_ind==i+1)[0]
            if len(ind)>0:
                print 'number of groups:', len(ind)
                ids = GC['GROUP_ID'][centrals[ind]]
                galaxy_members = np.in1d(GC['GROUP_ID'],ids) #galaxies in the mass bin
                galaxy_members = np.where(galaxy_members)[0] #indicies of galaxies
                print 'number of galaxies:', len(galaxy_members)
                satellite_members = np.where(GC['RPROJ'][galaxy_members]>0)[0] #satellites
                satellite_members = galaxy_members[satellite_members] #indices of satellites
                central_members = np.where(GC['RPROJ'][galaxy_members]==0)[0] #centrals
                central_members = galaxy_members[central_members] #indices of centrals
                print 'number of centrals:', len(central_members)
                print 'number of satellites:', len(satellite_members)
                print 'check:',  len(central_members) + len(satellite_members) == len(galaxy_members)
                blue_central_members = np.where(lookup[central_members]==True)[0] #blue centrals
                blue_central_members = central_members[blue_central_members] #indicies of blue centrals
                red_central_members = np.where(lookup[central_members]==False)[0] #red centrals
                red_central_members = central_members[red_central_members] #indicies of red centrals

                print 'number of blue centrals:', len(blue_central_members)
                print 'number of red centrals:', len(red_central_members) 
                print 'check:', len(blue_central_members)+len(red_central_members) == len(central_members)  
            
                blue_central_satellites = np.in1d(GC['GROUP_ID'][satellite_members],GC['GROUP_ID'][blue_central_members])
                blue_central_satellites = np.where(blue_central_satellites)[0]
                blue_central_satellites = satellite_members[blue_central_satellites]
                red_central_satellites = np.in1d(GC['GROUP_ID'][satellite_members],GC['GROUP_ID'][red_central_members])
                red_central_satellites = np.where(red_central_satellites)[0]
                red_central_satellites = satellite_members[red_central_satellites]
            
                print 'number of blue central satellites:', len(blue_central_satellites)
                print 'number of red central satellites:', len(red_central_satellites)
                print 'check:', len(blue_central_satellites) + len(red_central_satellites) == len(satellite_members)

                blue_central_blue_satellites = np.where(lookup[blue_central_satellites]==True)[0]
                blue_central_blue_satellites = blue_central_satellites[blue_central_blue_satellites]
                blue_central_red_satellites = np.where(lookup[blue_central_satellites]==False)[0]
                blue_central_red_satellites = blue_central_satellites[blue_central_red_satellites]
                red_central_blue_satellites = np.where(lookup[red_central_satellites]==True)[0]
                red_central_blue_satellites = red_central_satellites[red_central_blue_satellites]
                red_central_red_satellites = np.where(lookup[red_central_satellites]==False)[0]
                red_central_red_satellites = red_central_satellites[red_central_red_satellites]

                N_blue_central_blue_satellites  = float(len(blue_central_blue_satellites))
                N_blue_central_red_satellites = float(len(blue_central_red_satellites))
                N_red_central_blue_satellites  = float(len(red_central_blue_satellites))
                N_red_central_red_satellites = float(len(red_central_red_satellites))
            
                print 'check:', N_blue_central_blue_satellites+N_blue_central_red_satellites==len(blue_central_satellites)
                print 'check:', N_red_central_blue_satellites+N_red_central_red_satellites==len(red_central_satellites)

                if len(blue_central_satellites)>0: 
                    f_cen_blue[boot,i] = N_blue_central_blue_satellites / \
                             (N_blue_central_blue_satellites + N_blue_central_red_satellites)
                    print 'f_cen_blue:', f_cen_blue[boot,i]
                if len(red_central_satellites)>0:
                    f_cen_red[boot,i] = N_red_central_blue_satellites / \
                            (N_red_central_blue_satellites + N_red_central_red_satellites)
                    print 'f_cen_red:', f_cen_red[boot,i]

                #number satellites in the bin
                N_cen_blue[boot,i] = float(len(blue_central_satellites))
                N_cen_red[boot,i] = float(len(red_central_satellites))
                #number of centrals in the bin
                N_groups_blue[boot,i] = float(len(blue_central_members))
                N_groups_red[boot,i] = float(len(red_central_members))

    f_blue = np.nanmean(f_cen_blue, axis=0)
    f_red = np.nanmean(f_cen_red, axis=0)
    N_blue = np.nanmean(N_cen_blue, axis=0)
    N_red = np.nanmean(N_cen_red, axis=0)
    err_blue = np.nanstd(f_cen_blue, axis=0)
    err_red = np.nanstd(f_cen_red, axis=0)
    
    ax=axes[2]
    selection = np.where( N_blue > 50)[0]
    p1=ax.errorbar(mass_bins[0:-1][selection],f_blue[selection],yerr=err_blue[selection], fmt='o', color='blue', mec='none', ms=3)
    selection = np.where( N_red > 50)[0]
    p2=ax.errorbar(mass_bins[0:-1][selection],f_red[selection], yerr=err_red[selection], fmt='o', color='red', mec='none', ms=3)


    ######################################################################################
    ######################################################################################


    catalogue = catalogue_2

    print 'opening mock catalogue:', catalogue+'.hdf5'
    #open catalogue
    filepath_cat = cu.get_output_path() + 'processed_data/hearin_mocks/custom_catalogues/'
    f1 = h5py.File(filepath_cat+catalogue+'.hdf5', 'r') #open catalogue file
    GC = f1.get(catalogue)

    #galaxy color
    h=1.0
    color = GC['g-r']
    #LHS = 0.7 - 0.032*(GC['M_r,0.1']-5.0*np.log10(h)+16.5) #Weinmann 2006
    LHS = 0.21-0.03*GC['M_r,0.1']
    blue = np.where(color<LHS)[0] #indices of blue galaxies
    red = np.where(color>LHS)[0] #indicies of red galaxies
    lookup = color<LHS #true if blue, false if red
    flip = (lookup == False)

    centrals = np.where(GC['ID_host'] == -1)[0] #central galaxies
    N_groups = len(centrals)
    
    mass_bins = np.arange(10,15.2,0.2) #log mass bins
    mass_hist, bins = np.histogram(GC['M_host'][centrals], bins=mass_bins) #group histogram by log(mass)
    mass_bin_ind = np.digitize(GC['M_host'][centrals], bins=mass_bins) #indices of groups in log(mass) bins
    selection = np.where(lookup==True)[0]
    selection = np.intersect1d(selection,centrals)
    mass_hist_blue, bins = np.histogram(GC['M_host'][selection], bins=mass_bins) #group histogram by log(mass)
    mass_bin_ind_blue = np.digitize(GC['M_host'][selection], bins=mass_bins) #indices of groups in log(mass) bins
    selection = np.where(lookup==False)[0]
    selection = np.intersect1d(selection,centrals)
    mass_hist_red, bins = np.histogram(GC['M_host'][selection], bins=mass_bins) #group histogram by log(mass)
    mass_bin_ind_red = np.digitize(GC['M_host'][selection], bins=mass_bins) #indices of groups in log(mass) bins
    width = 1.0 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    #normalize the hists
    mass_hist_red = mass_hist_red/float(N_groups)
    mass_hist_blue = mass_hist_blue/float(N_groups)

    f_cen_blue   = np.zeros(len(mass_bins)-1) #fraction of blue satellites for blue centrals
    f_cen_red    = np.zeros(len(mass_bins)-1) #fraction of blue satellites for red centrals
    err_cen_blue = np.zeros(len(mass_bins)-1) #poisson error on blue fraction for blue centrals
    err_cen_red  = np.zeros(len(mass_bins)-1) #poisson error on blue fraction for red centrals
    N_cen_blue   = np.zeros(len(mass_bins)-1) #number of satellites for blue centrals
    N_cen_red    = np.zeros(len(mass_bins)-1) #number of satellites for red centrals
    N_groups_blue= np.zeros(len(mass_bins)-1) #number of satellites for red centrals
    N_groups_red = np.zeros(len(mass_bins)-1) #number of satellites for red centrals
    for i in range(0,len(mass_bins)-1):
        print i, mass_bins[i], mass_bins[i+1]
        ind = np.where(mass_bin_ind==i+1)[0]
        if len(ind)>0:
            print 'number of groups:', len(ind)
            ids = GC['ID_halo'][centrals[ind]]
            sat_galaxy_members = np.in1d(GC['ID_host'],ids) #satellite galaxies in the mass bin
            sat_galaxy_members = np.where(sat_galaxy_members)[0] #indicies of galaxies
            cen_galaxy_members = np.in1d(GC['ID_halo'],ids) #central galaxies in the mass bin
            cen_galaxy_members = np.where(cen_galaxy_members)[0] #indicies of galaxies
            galaxy_members = np.hstack((sat_galaxy_members,cen_galaxy_members))
            print 'number of galaxies:', len(galaxy_members)
            satellite_members = np.where(GC['ID_host'][galaxy_members]!=-1)[0] #satellites
            satellite_members = galaxy_members[satellite_members] #indices of satellites
            central_members = np.where(GC['ID_host'][galaxy_members]==-1)[0] #centrals
            central_members = galaxy_members[central_members] #indices of centrals
            print 'number of centrals:', len(central_members)
            print 'number of satellites:', len(satellite_members)
            print 'check:',  len(central_members) + len(satellite_members) == len(galaxy_members)
            blue_central_members = np.where(lookup[central_members]==True)[0] #blue centrals
            blue_central_members = central_members[blue_central_members] #indicies of blue centrals
            red_central_members = np.where(lookup[central_members]==False)[0] #red centrals
            red_central_members = central_members[red_central_members] #indicies of red centrals

            print 'number of blue centrals:', len(blue_central_members)
            print 'number of red centrals:', len(red_central_members) 
            print 'check:', len(blue_central_members)+len(red_central_members) == len(central_members)  
            
            blue_central_satellites = np.in1d(GC['ID_host'][satellite_members],GC['ID_halo'][blue_central_members])
            blue_central_satellites = np.where(blue_central_satellites)[0]
            blue_central_satellites = satellite_members[blue_central_satellites]
            red_central_satellites = np.in1d(GC['ID_host'][satellite_members],GC['ID_halo'][red_central_members])
            red_central_satellites = np.where(red_central_satellites)[0]
            red_central_satellites = satellite_members[red_central_satellites]
            
            print 'number of blue central satellites:', len(blue_central_satellites)
            print 'number of red central satellites:', len(red_central_satellites)
            print 'check:', len(blue_central_satellites) + len(red_central_satellites) == len(satellite_members)

            blue_central_blue_satellites = np.where(lookup[blue_central_satellites]==True)[0]
            blue_central_blue_satellites = blue_central_satellites[blue_central_blue_satellites]
            blue_central_red_satellites = np.where(lookup[blue_central_satellites]==False)[0]
            blue_central_red_satellites = blue_central_satellites[blue_central_red_satellites]
            red_central_blue_satellites = np.where(lookup[red_central_satellites]==True)[0]
            red_central_blue_satellites = red_central_satellites[red_central_blue_satellites]
            red_central_red_satellites = np.where(lookup[red_central_satellites]==False)[0]
            red_central_red_satellites = red_central_satellites[red_central_red_satellites]

            N_blue_central_blue_satellites  = float(len(blue_central_blue_satellites))
            N_blue_central_red_satellites = float(len(blue_central_red_satellites))
            N_red_central_blue_satellites  = float(len(red_central_blue_satellites))
            N_red_central_red_satellites = float(len(red_central_red_satellites))
            
            print 'check:', N_blue_central_blue_satellites+N_blue_central_red_satellites==len(blue_central_satellites)
            print 'check:', N_red_central_blue_satellites+N_red_central_red_satellites==len(red_central_satellites)

            if len(blue_central_satellites)>0: 
                f_cen_blue[i] = N_blue_central_blue_satellites / \
                             (N_blue_central_blue_satellites + N_blue_central_red_satellites)
                print 'f_cen_blue:', f_cen_blue[i]
            if len(red_central_satellites)>0:
                f_cen_red[i] = N_red_central_blue_satellites / \
                            (N_red_central_blue_satellites + N_red_central_red_satellites)
                print 'f_cen_red:', f_cen_red[i]

            #number satellites in the bin
            N_cen_blue[i] = float(len(blue_central_satellites))
            N_cen_red[i] = float(len(red_central_satellites))
            #number of centrals in the bin
            N_groups_blue[i] = float(len(blue_central_members))
            N_groups_red[i] = float(len(red_central_members))
   

    ax=axes[3]
    selection = np.where( N_cen_blue > 50)[0]
    p3, = ax.plot(mass_bins[0:-1][selection],f_cen_blue[selection], color='blue', alpha=1)
    selection = np.where( N_cen_red > 50)[0]
    p4, = ax.plot(mass_bins[0:-1][selection],f_cen_red[selection], color='red', alpha=1)
    ax=axes[4]
    selection = np.where( N_cen_blue > 50)[0]
    p3, = ax.plot(mass_bins[0:-1][selection],f_cen_blue[selection], color='blue', alpha=1)
    selection = np.where( N_cen_red > 50)[0]
    p4, = ax.plot(mass_bins[0:-1][selection],f_cen_red[selection], color='red', alpha=1)
    ax=axes[5]
    selection = np.where( N_cen_blue > 50)[0]
    p3, = ax.plot(mass_bins[0:-1][selection],f_cen_blue[selection], color='blue', alpha=1)
    selection = np.where( N_cen_red > 50)[0]
    p4, = ax.plot(mass_bins[0:-1][selection],f_cen_red[selection], color='red', alpha=1)
    
    #################################################################################################

    group_cat = 'berlind_mocks'

    if group_cat == 'tinker_mocks':
        filepath_cat = cu.get_output_path() + 'processed_data/tinker_groupcat/mock_runs/4th_run/custom_catalogues/'
        savepath = cu.get_output_path() + 'analysis/tinker_groupcat/'
        plotpath = cu.get_plot_path() + 'analysis/tinker_groupcat/'
        group_catalogue = catalogue+'_clf_groups_M19'
    if group_cat == 'berlind_mocks':
        filepath_cat = cu.get_output_path() + 'processed_data/berlind_groupcat/mock_runs/4th_run/custom_catalogues/'
        savepath = cu.get_output_path() + 'analysis/berlind_groupcat/'
        plotpath = cu.get_plot_path() + 'analysis/berlind_groupcat/'
        group_catalogue = catalogue+'_groups'
    if group_cat == 'yang_mocks':
        filepath_cat = cu.get_output_path() + 'processed_data/yang_groupcat/mock_runs/4th_run/custom_catalogues/'
        savepath = cu.get_output_path() + 'analysis/yang_groupcat/'
        plotpath = cu.get_plot_path() + 'analysis/yang_groupcat/'
        group_catalogue = catalogue+'_groups'

    mass_bins = np.arange(10,15.2,0.2) #log mass bins
    f_cen_blue   = np.zeros((N_boots,len(mass_bins)-1)) #fraction of blue satellites for blue centrals
    f_cen_red    = np.zeros((N_boots,len(mass_bins)-1)) #fraction of blue satellites for red centrals
    N_cen_blue   = np.zeros((N_boots,len(mass_bins)-1)) #number of satellites for blue centrals
    N_cen_red    = np.zeros((N_boots,len(mass_bins)-1)) #number of satellites for red centrals
    N_groups_blue= np.zeros((N_boots,len(mass_bins)-1)) #number of satellites for red centrals
    N_groups_red = np.zeros((N_boots,len(mass_bins)-1)) #number of satellites for red centrals

    for boot in range(0,N_boots):

        print 'opening group catalogue:', catalogue
        #open catalogue
        filepath = filepath_cat+'bootstraps/'
        catalogue_1 = group_catalogue+'_'+str(boot)
        print catalogue_1
        f =  h5py.File(filepath+catalogue_1+'.hdf5', 'r') #open catalogue file
        GC = f.get(catalogue_1)
        print len(GC)

        #do we have a measure of M_u?
        good = np.where(GC['M_g,0.1']!=-99.9)[0]
        bad  = np.where(GC['M_g,0.1']==-99.9)[0]
        print 'number of galaxies without M_u:', len(bad)    

        #galaxy color
        h=1.0
        color = GC['M_g,0.1'] - GC['M_r,0.1']
        #LHS = 0.7 - 0.032*(GC['M_r,0.1']-5.0*np.log10(h)+16.5) #Weinmann 2006
        LHS = 0.21-0.03*GC['M_r,0.1']
        blue = np.where(color<LHS)[0] #indices of blue galaxies
        red = np.where(color>LHS)[0] #indicies of red galaxies
        lookup = color<LHS #true if blue, false if red
        flip = (lookup == False)

        print 'number of blue galaxies:', len(blue)
        print 'number of red galaxies:', len(red)

        centrals = np.where(GC['RPROJ'][good] == 0)[0] #central galaxies
        centrals = good[centrals]
        bad_centrals = np.where(GC['RPROJ'][bad] == 0)[0] #central galaxies with ssfr measured
        bad_centrals = bad[bad_centrals]
        N_groups = len(centrals)+len(bad_centrals)
    
        mass_hist, bins = np.histogram(GC['MGROUP'][centrals], bins=mass_bins) #group histogram by log(mass)
        mass_bin_ind = np.digitize(GC['MGROUP'][centrals], bins=mass_bins) #indices of groups in log(mass) bins
        selection = np.where(lookup==True)[0]
        selection = np.intersect1d(selection,centrals)
        mass_hist_blue, bins = np.histogram(GC['MGROUP'][selection], bins=mass_bins) #group histogram by log(mass)
        mass_bin_ind_blue = np.digitize(GC['MGROUP'][selection], bins=mass_bins) #indices of groups in log(mass) bins
        selection = np.where(lookup==False)[0]
        selection = np.intersect1d(selection,centrals)
        mass_hist_red, bins = np.histogram(GC['MGROUP'][selection], bins=mass_bins) #group histogram by log(mass)
        mass_bin_ind_red = np.digitize(GC['MGROUP'][selection], bins=mass_bins) #indices of groups in log(mass) bins
        selection = bad_centrals
        mass_hist_grey, bins = np.histogram(GC['MGROUP'][selection], bins=mass_bins) #group histogram by log(mass)
        if len(bad_centrals)>0:
            mass_bin_ind_grey = np.digitize(GC['MGROUP'][selection], bins=mass_bins) #indices of groups in log(mass) bins
        width = 1.0 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        #normalize the hists
        mass_hist_grey = mass_hist_grey/float(N_groups)
        mass_hist_red = mass_hist_red/float(N_groups)
        mass_hist_blue = mass_hist_blue/float(N_groups)

    
        for i in range(0,len(mass_bins)-1):
            print i, mass_bins[i], mass_bins[i+1]
            ind = np.where(mass_bin_ind==i+1)[0]
            if len(ind)>0:
                print 'number of groups:', len(ind)
                ids = GC['GROUP_ID'][centrals[ind]]
                galaxy_members = np.in1d(GC['GROUP_ID'],ids) #galaxies in the mass bin
                galaxy_members = np.where(galaxy_members)[0] #indicies of galaxies
                print 'number of galaxies:', len(galaxy_members)
                satellite_members = np.where(GC['RPROJ'][galaxy_members]>0)[0] #satellites
                satellite_members = galaxy_members[satellite_members] #indices of satellites
                central_members = np.where(GC['RPROJ'][galaxy_members]==0)[0] #centrals
                central_members = galaxy_members[central_members] #indices of centrals
                print 'number of centrals:', len(central_members)
                print 'number of satellites:', len(satellite_members)
                print 'check:',  len(central_members) + len(satellite_members) == len(galaxy_members)
                blue_central_members = np.where(lookup[central_members]==True)[0] #blue centrals
                blue_central_members = central_members[blue_central_members] #indicies of blue centrals
                red_central_members = np.where(lookup[central_members]==False)[0] #red centrals
                red_central_members = central_members[red_central_members] #indicies of red centrals

                print 'number of blue centrals:', len(blue_central_members)
                print 'number of red centrals:', len(red_central_members) 
                print 'check:', len(blue_central_members)+len(red_central_members) == len(central_members)  
            
                blue_central_satellites = np.in1d(GC['GROUP_ID'][satellite_members],GC['GROUP_ID'][blue_central_members])
                blue_central_satellites = np.where(blue_central_satellites)[0]
                blue_central_satellites = satellite_members[blue_central_satellites]
                red_central_satellites = np.in1d(GC['GROUP_ID'][satellite_members],GC['GROUP_ID'][red_central_members])
                red_central_satellites = np.where(red_central_satellites)[0]
                red_central_satellites = satellite_members[red_central_satellites]
            
                print 'number of blue central satellites:', len(blue_central_satellites)
                print 'number of red central satellites:', len(red_central_satellites)
                print 'check:', len(blue_central_satellites) + len(red_central_satellites) == len(satellite_members)

                blue_central_blue_satellites = np.where(lookup[blue_central_satellites]==True)[0]
                blue_central_blue_satellites = blue_central_satellites[blue_central_blue_satellites]
                blue_central_red_satellites = np.where(lookup[blue_central_satellites]==False)[0]
                blue_central_red_satellites = blue_central_satellites[blue_central_red_satellites]
                red_central_blue_satellites = np.where(lookup[red_central_satellites]==True)[0]
                red_central_blue_satellites = red_central_satellites[red_central_blue_satellites]
                red_central_red_satellites = np.where(lookup[red_central_satellites]==False)[0]
                red_central_red_satellites = red_central_satellites[red_central_red_satellites]

                N_blue_central_blue_satellites  = float(len(blue_central_blue_satellites))
                N_blue_central_red_satellites = float(len(blue_central_red_satellites))
                N_red_central_blue_satellites  = float(len(red_central_blue_satellites))
                N_red_central_red_satellites = float(len(red_central_red_satellites))
            
                print 'check:', N_blue_central_blue_satellites+N_blue_central_red_satellites==len(blue_central_satellites)
                print 'check:', N_red_central_blue_satellites+N_red_central_red_satellites==len(red_central_satellites)

                if len(blue_central_satellites)>0: 
                    f_cen_blue[boot,i] = N_blue_central_blue_satellites / \
                             (N_blue_central_blue_satellites + N_blue_central_red_satellites)
                    print 'f_cen_blue:', f_cen_blue[boot,i]
                if len(red_central_satellites)>0:
                    f_cen_red[boot,i] = N_red_central_blue_satellites / \
                            (N_red_central_blue_satellites + N_red_central_red_satellites)
                    print 'f_cen_red:', f_cen_red[boot,i]

                #number satellites in the bin
                N_cen_blue[boot,i] = float(len(blue_central_satellites))
                N_cen_red[boot,i] = float(len(red_central_satellites))
                #number of centrals in the bin
                N_groups_blue[boot,i] = float(len(blue_central_members))
                N_groups_red[boot,i] = float(len(red_central_members))

    f_blue = np.nanmean(f_cen_blue, axis=0)
    f_red = np.nanmean(f_cen_red, axis=0)
    N_blue = np.nanmean(N_cen_blue, axis=0)
    N_red = np.nanmean(N_cen_red, axis=0)
    err_blue = np.nanstd(f_cen_blue, axis=0)
    err_red = np.nanstd(f_cen_red, axis=0)
    
    ax=axes[3]
    selection = np.where( N_blue > 50)[0]
    p1=ax.errorbar(mass_bins[0:-1][selection],f_blue[selection],yerr=err_blue[selection], fmt='o', color='blue', mec='none', ms=3)
    selection = np.where( N_red > 50)[0]
    p2=ax.errorbar(mass_bins[0:-1][selection],f_red[selection], yerr=err_red[selection], fmt='o', color='red', mec='none', ms=3)


    ######################################################################################

    group_cat = 'tinker_mocks'

    if group_cat == 'tinker_mocks':
        filepath_cat = cu.get_output_path() + 'processed_data/tinker_groupcat/mock_runs/4th_run/custom_catalogues/'
        savepath = cu.get_output_path() + 'analysis/tinker_groupcat/'
        plotpath = cu.get_plot_path() + 'analysis/tinker_groupcat/'
        group_catalogue = catalogue+'_clf_groups_M19'
    if group_cat == 'berlind_mocks':
        filepath_cat = cu.get_output_path() + 'processed_data/berlind_groupcat/mock_runs/4th_run/custom_catalogues/'
        savepath = cu.get_output_path() + 'analysis/berlind_groupcat/'
        plotpath = cu.get_plot_path() + 'analysis/berlind_groupcat/'
        group_catalogue = catalogue+'_groups'
    if group_cat == 'yang_mocks':
        filepath_cat = cu.get_output_path() + 'processed_data/yang_groupcat/mock_runs/4th_run/custom_catalogues/'
        savepath = cu.get_output_path() + 'analysis/yang_groupcat/'
        plotpath = cu.get_plot_path() + 'analysis/yang_groupcat/'
        group_catalogue = catalogue+'_groups'

    mass_bins = np.arange(10,15.2,0.2) #log mass bins
    f_cen_blue   = np.zeros((N_boots,len(mass_bins)-1)) #fraction of blue satellites for blue centrals
    f_cen_red    = np.zeros((N_boots,len(mass_bins)-1)) #fraction of blue satellites for red centrals
    N_cen_blue   = np.zeros((N_boots,len(mass_bins)-1)) #number of satellites for blue centrals
    N_cen_red    = np.zeros((N_boots,len(mass_bins)-1)) #number of satellites for red centrals
    N_groups_blue= np.zeros((N_boots,len(mass_bins)-1)) #number of satellites for red centrals
    N_groups_red = np.zeros((N_boots,len(mass_bins)-1)) #number of satellites for red centrals

    for boot in range(0,N_boots):

        print 'opening group catalogue:', catalogue
        #open catalogue
        filepath = filepath_cat+'bootstraps/'
        catalogue_1 = group_catalogue+'_'+str(boot)
        print catalogue_1
        f =  h5py.File(filepath+catalogue_1+'.hdf5', 'r') #open catalogue file
        GC = f.get(catalogue_1)
        print len(GC)

        #do we have a measure of M_u?
        good = np.where(GC['M_g,0.1']!=-99.9)[0]
        bad  = np.where(GC['M_g,0.1']==-99.9)[0]
        print 'number of galaxies without M_u:', len(bad)    

        #galaxy color
        h=1.0
        color = GC['M_g,0.1'] - GC['M_r,0.1']
        #LHS = 0.7 - 0.032*(GC['M_r,0.1']-5.0*np.log10(h)+16.5) #Weinmann 2006
        LHS = 0.21-0.03*GC['M_r,0.1']
        blue = np.where(color<LHS)[0] #indices of blue galaxies
        red = np.where(color>LHS)[0] #indicies of red galaxies
        lookup = color<LHS #true if blue, false if red
        flip = (lookup == False)

        print 'number of blue galaxies:', len(blue)
        print 'number of red galaxies:', len(red)

        centrals = np.where(GC['RPROJ'][good] == 0)[0] #central galaxies
        centrals = good[centrals]
        bad_centrals = np.where(GC['RPROJ'][bad] == 0)[0] #central galaxies with ssfr measured
        bad_centrals = bad[bad_centrals]
        N_groups = len(centrals)+len(bad_centrals)
    
        mass_hist, bins = np.histogram(GC['MGROUP'][centrals], bins=mass_bins) #group histogram by log(mass)
        mass_bin_ind = np.digitize(GC['MGROUP'][centrals], bins=mass_bins) #indices of groups in log(mass) bins
        selection = np.where(lookup==True)[0]
        selection = np.intersect1d(selection,centrals)
        mass_hist_blue, bins = np.histogram(GC['MGROUP'][selection], bins=mass_bins) #group histogram by log(mass)
        mass_bin_ind_blue = np.digitize(GC['MGROUP'][selection], bins=mass_bins) #indices of groups in log(mass) bins
        selection = np.where(lookup==False)[0]
        selection = np.intersect1d(selection,centrals)
        mass_hist_red, bins = np.histogram(GC['MGROUP'][selection], bins=mass_bins) #group histogram by log(mass)
        mass_bin_ind_red = np.digitize(GC['MGROUP'][selection], bins=mass_bins) #indices of groups in log(mass) bins
        selection = bad_centrals
        mass_hist_grey, bins = np.histogram(GC['MGROUP'][selection], bins=mass_bins) #group histogram by log(mass)
        if len(bad_centrals)>0:
            mass_bin_ind_grey = np.digitize(GC['MGROUP'][selection], bins=mass_bins) #indices of groups in log(mass) bins
        width = 1.0 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        #normalize the hists
        mass_hist_grey = mass_hist_grey/float(N_groups)
        mass_hist_red = mass_hist_red/float(N_groups)
        mass_hist_blue = mass_hist_blue/float(N_groups)

    
        for i in range(0,len(mass_bins)-1):
            print i, mass_bins[i], mass_bins[i+1]
            ind = np.where(mass_bin_ind==i+1)[0]
            if len(ind)>0:
                print 'number of groups:', len(ind)
                ids = GC['GROUP_ID'][centrals[ind]]
                galaxy_members = np.in1d(GC['GROUP_ID'],ids) #galaxies in the mass bin
                galaxy_members = np.where(galaxy_members)[0] #indicies of galaxies
                print 'number of galaxies:', len(galaxy_members)
                satellite_members = np.where(GC['RPROJ'][galaxy_members]>0)[0] #satellites
                satellite_members = galaxy_members[satellite_members] #indices of satellites
                central_members = np.where(GC['RPROJ'][galaxy_members]==0)[0] #centrals
                central_members = galaxy_members[central_members] #indices of centrals
                print 'number of centrals:', len(central_members)
                print 'number of satellites:', len(satellite_members)
                print 'check:',  len(central_members) + len(satellite_members) == len(galaxy_members)
                blue_central_members = np.where(lookup[central_members]==True)[0] #blue centrals
                blue_central_members = central_members[blue_central_members] #indicies of blue centrals
                red_central_members = np.where(lookup[central_members]==False)[0] #red centrals
                red_central_members = central_members[red_central_members] #indicies of red centrals

                print 'number of blue centrals:', len(blue_central_members)
                print 'number of red centrals:', len(red_central_members) 
                print 'check:', len(blue_central_members)+len(red_central_members) == len(central_members)  
            
                blue_central_satellites = np.in1d(GC['GROUP_ID'][satellite_members],GC['GROUP_ID'][blue_central_members])
                blue_central_satellites = np.where(blue_central_satellites)[0]
                blue_central_satellites = satellite_members[blue_central_satellites]
                red_central_satellites = np.in1d(GC['GROUP_ID'][satellite_members],GC['GROUP_ID'][red_central_members])
                red_central_satellites = np.where(red_central_satellites)[0]
                red_central_satellites = satellite_members[red_central_satellites]
            
                print 'number of blue central satellites:', len(blue_central_satellites)
                print 'number of red central satellites:', len(red_central_satellites)
                print 'check:', len(blue_central_satellites) + len(red_central_satellites) == len(satellite_members)

                blue_central_blue_satellites = np.where(lookup[blue_central_satellites]==True)[0]
                blue_central_blue_satellites = blue_central_satellites[blue_central_blue_satellites]
                blue_central_red_satellites = np.where(lookup[blue_central_satellites]==False)[0]
                blue_central_red_satellites = blue_central_satellites[blue_central_red_satellites]
                red_central_blue_satellites = np.where(lookup[red_central_satellites]==True)[0]
                red_central_blue_satellites = red_central_satellites[red_central_blue_satellites]
                red_central_red_satellites = np.where(lookup[red_central_satellites]==False)[0]
                red_central_red_satellites = red_central_satellites[red_central_red_satellites]

                N_blue_central_blue_satellites  = float(len(blue_central_blue_satellites))
                N_blue_central_red_satellites = float(len(blue_central_red_satellites))
                N_red_central_blue_satellites  = float(len(red_central_blue_satellites))
                N_red_central_red_satellites = float(len(red_central_red_satellites))
            
                print 'check:', N_blue_central_blue_satellites+N_blue_central_red_satellites==len(blue_central_satellites)
                print 'check:', N_red_central_blue_satellites+N_red_central_red_satellites==len(red_central_satellites)

                if len(blue_central_satellites)>0: 
                    f_cen_blue[boot,i] = N_blue_central_blue_satellites / \
                             (N_blue_central_blue_satellites + N_blue_central_red_satellites)
                    print 'f_cen_blue:', f_cen_blue[boot,i]
                if len(red_central_satellites)>0:
                    f_cen_red[boot,i] = N_red_central_blue_satellites / \
                            (N_red_central_blue_satellites + N_red_central_red_satellites)
                    print 'f_cen_red:', f_cen_red[boot,i]

                #number satellites in the bin
                N_cen_blue[boot,i] = float(len(blue_central_satellites))
                N_cen_red[boot,i] = float(len(red_central_satellites))
                #number of centrals in the bin
                N_groups_blue[boot,i] = float(len(blue_central_members))
                N_groups_red[boot,i] = float(len(red_central_members))

    f_blue = np.nanmean(f_cen_blue, axis=0)
    f_red = np.nanmean(f_cen_red, axis=0)
    N_blue = np.nanmean(N_cen_blue, axis=0)
    N_red = np.nanmean(N_cen_red, axis=0)
    err_blue = np.nanstd(f_cen_blue, axis=0)
    err_red = np.nanstd(f_cen_red, axis=0)
    
    ax=axes[4]
    selection = np.where( N_blue > 50)[0]
    p1=ax.errorbar(mass_bins[0:-1][selection],f_blue[selection],yerr=err_blue[selection], fmt='o', color='blue', mec='none', ms=3)
    selection = np.where( N_red > 50)[0]
    p2=ax.errorbar(mass_bins[0:-1][selection],f_red[selection], yerr=err_red[selection], fmt='o', color='red', mec='none', ms=3)


    ######################################################################################


    group_cat = 'yang_mocks'

    if group_cat == 'tinker_mocks':
        filepath_cat = cu.get_output_path() + 'processed_data/tinker_groupcat/mock_runs/4th_run/custom_catalogues/'
        savepath = cu.get_output_path() + 'analysis/tinker_groupcat/'
        plotpath = cu.get_plot_path() + 'analysis/tinker_groupcat/'
        group_catalogue = catalogue+'_clf_groups_M19'
    if group_cat == 'berlind_mocks':
        filepath_cat = cu.get_output_path() + 'processed_data/berlind_groupcat/mock_runs/4th_run/custom_catalogues/'
        savepath = cu.get_output_path() + 'analysis/berlind_groupcat/'
        plotpath = cu.get_plot_path() + 'analysis/berlind_groupcat/'
        group_catalogue = catalogue+'_groups'
    if group_cat == 'yang_mocks':
        filepath_cat = cu.get_output_path() + 'processed_data/yang_groupcat/mock_runs/4th_run/custom_catalogues/'
        savepath = cu.get_output_path() + 'analysis/yang_groupcat/'
        plotpath = cu.get_plot_path() + 'analysis/yang_groupcat/'
        group_catalogue = catalogue+'_groups'

    mass_bins = np.arange(10,15.2,0.2) #log mass bins
    f_cen_blue   = np.zeros((N_boots,len(mass_bins)-1)) #fraction of blue satellites for blue centrals
    f_cen_red    = np.zeros((N_boots,len(mass_bins)-1)) #fraction of blue satellites for red centrals
    N_cen_blue   = np.zeros((N_boots,len(mass_bins)-1)) #number of satellites for blue centrals
    N_cen_red    = np.zeros((N_boots,len(mass_bins)-1)) #number of satellites for red centrals
    N_groups_blue= np.zeros((N_boots,len(mass_bins)-1)) #number of satellites for red centrals
    N_groups_red = np.zeros((N_boots,len(mass_bins)-1)) #number of satellites for red centrals

    for boot in range(0,N_boots):

        print 'opening group catalogue:', catalogue
        #open catalogue
        filepath = filepath_cat+'bootstraps/'
        catalogue_1 = group_catalogue+'_'+str(boot)
        print catalogue_1
        f =  h5py.File(filepath+catalogue_1+'.hdf5', 'r') #open catalogue file
        GC = f.get(catalogue_1)
        print len(GC)

        #do we have a measure of M_u?
        good = np.where(GC['M_g,0.1']!=-99.9)[0]
        bad  = np.where(GC['M_g,0.1']==-99.9)[0]
        print 'number of galaxies without M_u:', len(bad)    

        #galaxy color
        h=1.0
        color = GC['M_g,0.1'] - GC['M_r,0.1']
        #LHS = 0.7 - 0.032*(GC['M_r,0.1']-5.0*np.log10(h)+16.5) #Weinmann 2006
        LHS = 0.21-0.03*GC['M_r,0.1']
        blue = np.where(color<LHS)[0] #indices of blue galaxies
        red = np.where(color>LHS)[0] #indicies of red galaxies
        lookup = color<LHS #true if blue, false if red
        flip = (lookup == False)

        print 'number of blue galaxies:', len(blue)
        print 'number of red galaxies:', len(red)

        centrals = np.where(GC['RPROJ'][good] == 0)[0] #central galaxies
        centrals = good[centrals]
        bad_centrals = np.where(GC['RPROJ'][bad] == 0)[0] #central galaxies with ssfr measured
        bad_centrals = bad[bad_centrals]
        N_groups = len(centrals)+len(bad_centrals)
    
        mass_hist, bins = np.histogram(GC['MGROUP'][centrals], bins=mass_bins) #group histogram by log(mass)
        mass_bin_ind = np.digitize(GC['MGROUP'][centrals], bins=mass_bins) #indices of groups in log(mass) bins
        selection = np.where(lookup==True)[0]
        selection = np.intersect1d(selection,centrals)
        mass_hist_blue, bins = np.histogram(GC['MGROUP'][selection], bins=mass_bins) #group histogram by log(mass)
        mass_bin_ind_blue = np.digitize(GC['MGROUP'][selection], bins=mass_bins) #indices of groups in log(mass) bins
        selection = np.where(lookup==False)[0]
        selection = np.intersect1d(selection,centrals)
        mass_hist_red, bins = np.histogram(GC['MGROUP'][selection], bins=mass_bins) #group histogram by log(mass)
        mass_bin_ind_red = np.digitize(GC['MGROUP'][selection], bins=mass_bins) #indices of groups in log(mass) bins
        selection = bad_centrals
        mass_hist_grey, bins = np.histogram(GC['MGROUP'][selection], bins=mass_bins) #group histogram by log(mass)
        if len(bad_centrals)>0:
            mass_bin_ind_grey = np.digitize(GC['MGROUP'][selection], bins=mass_bins) #indices of groups in log(mass) bins
        width = 1.0 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        #normalize the hists
        mass_hist_grey = mass_hist_grey/float(N_groups)
        mass_hist_red = mass_hist_red/float(N_groups)
        mass_hist_blue = mass_hist_blue/float(N_groups)

    
        for i in range(0,len(mass_bins)-1):
            print i, mass_bins[i], mass_bins[i+1]
            ind = np.where(mass_bin_ind==i+1)[0]
            if len(ind)>0:
                print 'number of groups:', len(ind)
                ids = GC['GROUP_ID'][centrals[ind]]
                galaxy_members = np.in1d(GC['GROUP_ID'],ids) #galaxies in the mass bin
                galaxy_members = np.where(galaxy_members)[0] #indicies of galaxies
                print 'number of galaxies:', len(galaxy_members)
                satellite_members = np.where(GC['RPROJ'][galaxy_members]>0)[0] #satellites
                satellite_members = galaxy_members[satellite_members] #indices of satellites
                central_members = np.where(GC['RPROJ'][galaxy_members]==0)[0] #centrals
                central_members = galaxy_members[central_members] #indices of centrals
                print 'number of centrals:', len(central_members)
                print 'number of satellites:', len(satellite_members)
                print 'check:',  len(central_members) + len(satellite_members) == len(galaxy_members)
                blue_central_members = np.where(lookup[central_members]==True)[0] #blue centrals
                blue_central_members = central_members[blue_central_members] #indicies of blue centrals
                red_central_members = np.where(lookup[central_members]==False)[0] #red centrals
                red_central_members = central_members[red_central_members] #indicies of red centrals

                print 'number of blue centrals:', len(blue_central_members)
                print 'number of red centrals:', len(red_central_members) 
                print 'check:', len(blue_central_members)+len(red_central_members) == len(central_members)  
            
                blue_central_satellites = np.in1d(GC['GROUP_ID'][satellite_members],GC['GROUP_ID'][blue_central_members])
                blue_central_satellites = np.where(blue_central_satellites)[0]
                blue_central_satellites = satellite_members[blue_central_satellites]
                red_central_satellites = np.in1d(GC['GROUP_ID'][satellite_members],GC['GROUP_ID'][red_central_members])
                red_central_satellites = np.where(red_central_satellites)[0]
                red_central_satellites = satellite_members[red_central_satellites]
            
                print 'number of blue central satellites:', len(blue_central_satellites)
                print 'number of red central satellites:', len(red_central_satellites)
                print 'check:', len(blue_central_satellites) + len(red_central_satellites) == len(satellite_members)

                blue_central_blue_satellites = np.where(lookup[blue_central_satellites]==True)[0]
                blue_central_blue_satellites = blue_central_satellites[blue_central_blue_satellites]
                blue_central_red_satellites = np.where(lookup[blue_central_satellites]==False)[0]
                blue_central_red_satellites = blue_central_satellites[blue_central_red_satellites]
                red_central_blue_satellites = np.where(lookup[red_central_satellites]==True)[0]
                red_central_blue_satellites = red_central_satellites[red_central_blue_satellites]
                red_central_red_satellites = np.where(lookup[red_central_satellites]==False)[0]
                red_central_red_satellites = red_central_satellites[red_central_red_satellites]

                N_blue_central_blue_satellites  = float(len(blue_central_blue_satellites))
                N_blue_central_red_satellites = float(len(blue_central_red_satellites))
                N_red_central_blue_satellites  = float(len(red_central_blue_satellites))
                N_red_central_red_satellites = float(len(red_central_red_satellites))
            
                print 'check:', N_blue_central_blue_satellites+N_blue_central_red_satellites==len(blue_central_satellites)
                print 'check:', N_red_central_blue_satellites+N_red_central_red_satellites==len(red_central_satellites)

                if len(blue_central_satellites)>0: 
                    f_cen_blue[boot,i] = N_blue_central_blue_satellites / \
                             (N_blue_central_blue_satellites + N_blue_central_red_satellites)
                    print 'f_cen_blue:', f_cen_blue[boot,i]
                if len(red_central_satellites)>0:
                    f_cen_red[boot,i] = N_red_central_blue_satellites / \
                            (N_red_central_blue_satellites + N_red_central_red_satellites)
                    print 'f_cen_red:', f_cen_red[boot,i]

                #number satellites in the bin
                N_cen_blue[boot,i] = float(len(blue_central_satellites))
                N_cen_red[boot,i] = float(len(red_central_satellites))
                #number of centrals in the bin
                N_groups_blue[boot,i] = float(len(blue_central_members))
                N_groups_red[boot,i] = float(len(red_central_members))

    f_blue = np.nanmean(f_cen_blue, axis=0)
    f_red = np.nanmean(f_cen_red, axis=0)
    N_blue = np.nanmean(N_cen_blue, axis=0)
    N_red = np.nanmean(N_cen_red, axis=0)
    err_blue = np.nanstd(f_cen_blue, axis=0)
    err_red = np.nanstd(f_cen_red, axis=0)
    
    ax=axes[5]
    selection = np.where( N_blue > 50)[0]
    p1=ax.errorbar(mass_bins[0:-1][selection],f_blue[selection],yerr=err_blue[selection], fmt='o', color='blue', mec='none', ms=3)
    selection = np.where( N_red > 50)[0]
    p2=ax.errorbar(mass_bins[0:-1][selection],f_red[selection], yerr=err_red[selection], fmt='o', color='red', mec='none', ms=3)


    ######################################################################################
    ######################################################################################

    plt.show(block=True)

    filename='conformity_'+catalogue+'.png'
    fig.savefig(plotpath+filename,dpi=400)

    print plotpath+filename


if __name__ == '__main__':
    main() 
