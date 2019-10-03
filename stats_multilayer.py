from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import fitsio
import desimodel.io
import desitarget.mtl
import desisim.quickcat
from astropy.io import fits
from astropy.table import Table, Column, vstack
import json
import shutil
import healpy
from desitarget.targetmask import desi_mask, obsconditions
from collections import Counter
import subprocess

def consolidate_favail(fba_files):
    # getting all the targetids of the assigned fibers
    print('reading individual fiberassign files')
    favail = list()
    for i_tile, tile_file in enumerate(fba_files):
        if i_tile%50 ==0:
            print(i_tile)
        id_favail, header = fits.getdata(tile_file, 'FAVAIL', header=True)
        favail.extend(id_favail['TARGETID'])
    return list(set(favail))

def global_efficiency(targets, id_avail, zcat, target_class='QSO', zcat_spectype='QSO', z_max=None, z_min=None):
    ii_avail = np.in1d(targets['TARGETID'], id_avail)
    targets_avail = targets[ii_avail]

    if z_max is None and z_min is None:
        sub_zcat = zcat.copy()
    elif (z_min is not None) or (z_max is not None):
        if z_max is not None:
            sub_zcat = zcat[zcat['Z']<z_max]
        if z_min is not None:
            sub_zcat = zcat[zcat['Z']>z_min]
    else:
        print("Error")
        sub_zcat = None

    # input target consistent with target_class
    is_class = (targets_avail['DESI_TARGET'] & desi_mask.mask(target_class))!=0
    targets_avail_class = targets_avail[is_class]
    n_avail = len(targets_avail_class)

    # output in the redshift catalog consistent with truth_spectype
    sub_zcat_class = sub_zcat[sub_zcat['SPECTYPE']==zcat_spectype]
    
    # keep the elements in the zcat that correspond to the correct input target class
    id_intersection = np.in1d(sub_zcat_class['TARGETID'], targets_avail_class['TARGETID'])
    sub_zcat_class = sub_zcat_class[id_intersection]
    n_assigned = len(sub_zcat_class)

    nobs = dict()
    for i in range(10):
        nobs[i] = np.count_nonzero(sub_zcat_class['NUMOBS']==i)
    nobs[0] = (n_avail - n_assigned)

    print(target_class, zcat_spectype, n_assigned/n_avail, n_avail, n_assigned, nobs)
    
def gather_files(strategy_name, pass_names):
    os.makedirs('{}/fiberassign_full'.format(strategy_name), exist_ok=True)
    for p_name in pass_names:
        os.system('cp -v {}/fiberassign_{}/tile*.fits {}/fiberassign_full'.format(strategy_name, p_name, strategy_name))

#gather_files('strategy_A', ['gray', 'dark0', 'dark1', 'dark2_dark3'])
#gather_files('strategy_B', ['gray', 'dark0', 'dark1', 'dark2_dark3'])


def compute_efficiency(strategy_name, pass_names, targets_file, truth_file):
    gather_files(strategy_name, pass_names)
    zcat_file = '{}/zcat/{}_zcat.fits'.format(strategy_name, pass_names[-1])
    fba_path = '{}/fiberassign_full/tile-*fits'.format(strategy_name)
    
    print('Consolidating info from {}'.format(fba_path))
    fba_files= glob.glob(fba_path)
    favail = consolidate_favail(fba_files)
    
    print('reading zcat', zcat_file)
    zcat = Table.read(zcat_file)
    
    print('reading targets', targets_file)
    targets = Table.read(targets_file)
    
    print('reading truth', truth_file)
    truth = Table.read(truth_file)

    print('Sorting files')
    targets.sort(keys='TARGETID')
    zcat.sort(keys='TARGETID')
    truth.sort(keys='TARGETID')
    
targets_file =     "targets/subset_dr8_mtl_dark_gray_NGC.fits"
truth_file =     "targets/subset_truth_dr8_mtl_dark_gray_NGC.fits"
compute_efficiency('strategy_A', ['gray', 'dark0', 'dark1', 'dark2_dark3'], targets_file, truth_file)
