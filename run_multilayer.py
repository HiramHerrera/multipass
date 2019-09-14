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
import mtl as mymtl
import targets


from desitarget.targetmask import desi_mask, obsconditions
from collections import Counter
import subprocess

initial_mtl_file = "targets/subset_dr8_mtl_dark_gray_NGC.fits"
if not os.path.exists(initial_mtl_file):
    path_to_targets = '/project/projectdirs/desi/target/catalogs/dr8/0.31.1/targets/main/resolve/'
    target_files = glob.glob(os.path.join(path_to_targets, "targets*fits"))
    print(len(target_files))
    target_files.sort()
    print(target_files)
    
    data = fitsio.FITS(target_files[0], 'r')
    target_data = data[1].read(columns=['TARGETID', 'DESI_TARGET', 'MWS_TARGET', 'BGS_TARGET', 'SUBPRIORITY', 'NUMOBS_INIT', 'PRIORITY_INIT', 'RA', 'DEC', 'HPXPIXEL'])
    data.close()
    for i, i_name in enumerate(target_files[1:]):
        data = fitsio.FITS(i_name, 'r')
        tmp_data = data[1].read(columns=['TARGETID', 'DESI_TARGET', 'MWS_TARGET', 'BGS_TARGET', 'SUBPRIORITY', 'NUMOBS_INIT', 'PRIORITY_INIT', 'RA', 'DEC', 'HPXPIXEL'])
        target_data = np.hstack((target_data, tmp_data))
        data.close()
        print(i, len(target_files), len(tmp_data))
    #full_mtl = mymtl.make_mtl(target_data, bright=False)
    full_mtl = desitarget.mtl.make_mtl(target_data, 'DARK|GRAY')

    ii_mtl_dark = (full_mtl['OBSCONDITIONS'] & obsconditions.DARK)!=0
    ii_mtl_gray = (full_mtl['OBSCONDITIONS'] & obsconditions.GRAY)!=0
    ii_inside = (full_mtl['DEC']>-20)
    ii_south = (full_mtl['RA']<85) | (full_mtl['RA']>300)
    ii_north = (full_mtl['RA']>85) & (full_mtl['RA']<300) & (full_mtl['DEC']>-15)

    mtl_file = "targets/dr8_mtl_dark_gray_northern_cap.fits"
    full_mtl[(ii_mtl_dark | ii_mtl_gray)&ii_inside&ii_north].write(mtl_file, overwrite=True)
    
    mtl_data = Table.read(mtl_file)
    subset_ii = (mtl_data['RA']>155) & (mtl_data['RA']<185)
    subset_ii &= (mtl_data['DEC']>-5) & (mtl_data['DEC']<25)
    mtl_data[subset_ii].write(initial_mtl_file, overwrite=True)

initial_sky_file = "targets/subset_dr8_sky.fits"
if not os.path.exists(initial_sky_file):
    sky_data = Table.read("/project/projectdirs/desi/target/catalogs/dr8/0.31.0/skies/skies-dr8-0.31.0.fits")
    subset_ii = (sky_data['RA']>155) & (sky_data['RA']<185)
    subset_ii &= (sky_data['DEC']>-5) & (sky_data['DEC']<25)
    print('writing sky')
    sky_data[subset_ii].write(initial_sky_file, overwrite=True)
    print('done writing sky')
    
def assign_lya_qso(initial_mtl_file, pixweight_file):
    targets = Table.read(initial_mtl_file)

    pixweight, header = fits.getdata(pixweight_file, 'PIXWEIGHTS', header=True)
    hpxnside = header['HPXNSIDE']

    theta_w, phi_w = healpy.pix2ang(hpxnside, pixweight['HPXPIXEL'], nest=True)

    hpxnside_sample = 64 # pixel area on which we will sample the lyaqso
    npix_sample = healpy.nside2npix(hpxnside_sample)
    pixnumber_sample = healpy.ang2pix(hpxnside_sample, theta_w, phi_w, nest=True)

    subpixels = {} # store the pixels at the resolution in the input target catalog that are included within the pixels at the new resolution.
    for i in range(npix_sample):
        ii_sample = pixnumber_sample==i
        subpixels[i] =  healpy.ang2pix(hpxnside, theta_w[ii_sample], phi_w[ii_sample], nest=True)
    
    # redefine the covered area for the new pixels from the higher resolution FRACAREA
    covered_area = np.ones(npix_sample)
    for i in range(npix_sample):
        sum_weight = np.sum(pixweight['FRACAREA'][subpixels[i]])
        if sum_weight>0.0:
            covered_area[i] = np.sum(pixweight['FRACAREA'][subpixels[i]]**2)/np.sum(pixweight['FRACAREA'][subpixels[i]])
        else:
            covered_area[i] = 0.0

    theta_s, phi_s = healpy.pix2ang(hpxnside_sample, np.arange(npix_sample), nest=True)

    pixelarea_sample = healpy.pixelfunc.nside2pixarea(hpxnside_sample, degrees=True)
    n_lya_qso_in_pixel = np.int_(covered_area * 50 * pixelarea_sample)

    # compute angular coordinates from the targets
    targets_phi = np.deg2rad(targets['RA'])
    targets_theta = np.deg2rad(90.0-targets['DEC'])

    # find the pixnumber to which the target belongs (in the hpxnside_sample resolution)
    pixnumber_targets = healpy.ang2pix(hpxnside_sample, targets_theta, targets_phi, nest=True)

    
    # what target are QSOs?
    is_qso = (targets['DESI_TARGET'] & desi_mask.QSO)!=0

    # list of unique pixels covered by the targets
    pixnumber_target_list = list(set(pixnumber_targets)) # list of pixelsIDs covered by the targets in the new resolution

    n_qso_per_pixel_targets = np.zeros(len(pixnumber_target_list))
    for i in range(len(pixnumber_target_list)):
        ii_targets = is_qso & (pixnumber_targets==pixnumber_target_list[i])
        n_qso_per_pixel_targets[i] = np.count_nonzero(ii_targets)
    
    n_lya_desired_pixel_targets = np.random.poisson(n_lya_qso_in_pixel[pixnumber_target_list])

    # Generate the boolean array to determine whether a target is a lyaqso or not
    is_lya_qso = np.repeat(False, len(targets))
    n_targets = len(targets)
    target_ids = np.arange(n_targets)

    for i in range(len(pixnumber_target_list)):
        ii_targets = is_qso & (pixnumber_targets==pixnumber_target_list[i])
        n_qso_in_pixel = np.count_nonzero(ii_targets)
        n_lya_desired = n_lya_desired_pixel_targets[i]
        if n_lya_desired >= n_qso_in_pixel:
            is_lya_qso[ii_targets] = True
        else:
            #print(len(target_ids[ii_targets]), n_lya_desired)
            ii_lya_qso = np.random.choice(target_ids[ii_targets], n_lya_desired, replace=False)
            is_lya_qso[ii_lya_qso] = True
    return is_lya_qso

initial_truth_file = "targets/subset_truth_dr8_mtl_dark_gray_NGC.fits"
pixweight_file = "/project/projectdirs/desi/target/catalogs/dr8/0.31.1/pixweight/pixweight-dr8-0.31.1.fits"

if not os.path.exists(initial_truth_file):
    import desitarget.mock.mockmaker as mb
    from desitarget.targetmask import desi_mask, bgs_mask, mws_mask

    is_lya_qso = assign_lya_qso(initial_mtl_file, pixweight_file)
    
    targets = Table.read(initial_mtl_file)
    colnames = list(targets.dtype.names)
    print(colnames)
    nobj = len(targets)
    truth = mb.empty_truth_table(nobj=nobj)[0]
    print(truth.keys())

    for k in colnames:
        if k in truth.keys():
            print(k)
            truth[k][:] = targets[k][:]

    nothing = '          '
    truth['TEMPLATESUBTYPE'] = np.repeat(nothing, nobj)

    masks = ['MWS_ANY', 'BGS_ANY', 'STD_FAINT', 'STD_BRIGHT','ELG', 'LRG', 'QSO', ]
    dict_truespectype = {'BGS_ANY':'GALAXY', 'ELG':'GALAXY', 'LRG':'GALAXY', 'QSO':'QSO', 
                    'MWS_ANY':'STAR', 'STD_FAINT':'STAR', 'STD_BRIGHT':'STAR'}
    dict_truetemplatetype = {'BGS_ANY':'BGS', 'ELG':'ELG', 'LRG':'LRG', 'QSO':'QSO', 
                        'MWS_ANY':'STAR', 'STD_FAINT':'STAR', 'STD_BRIGHT':'STAR'}
    dict_truez = {'BGS_ANY':0.2, 'ELG':1.5, 'LRG':0.7, 'QSO':2.0, 
                        'MWS_ANY':0.0, 'STD_FAINT':0.0, 'STD_BRIGHT':0.0}

    for m in masks:
        istype = (targets['DESI_TARGET'] & desi_mask.mask(m))!=0
        print(m, np.count_nonzero(istype))
        truth['TRUESPECTYPE'][istype] = np.repeat(dict_truespectype[m], np.count_nonzero(istype))
        truth['TEMPLATETYPE'][istype] = np.repeat(dict_truetemplatetype[m], np.count_nonzero(istype))
        truth['MOCKID'][istype] = targets['TARGETID'][istype]
        truth['TRUEZ'][istype] = dict_truez[m]
        
    truth['TRUEZ'][is_lya_qso] = 3.0

    # Check that all targets have been assigned to a class
    iii = truth['MOCKID']==0
    assert np.count_nonzero(iii)==0
    
    print('writing truth')
    truth.write(initial_truth_file, overwrite=True)
    print('done truth')
    
def prepare_tiles():
    tiles = Table(desimodel.io.load_tiles())

    ii_tiles = tiles['PROGRAM'] != 'BRIGHT'
    ii_tiles &= tiles['RA'] > 160 
    ii_tiles &= tiles['RA'] < 180
    ii_tiles &= tiles['DEC'] > 0
    ii_tiles &= tiles['DEC'] < 20

    tilefile = 'footprint/subset_tiles.fits'
    tiles[ii_tiles].write(tilefile, overwrite='True')
    tiles = Table.read(tilefile)

    ii_gray = tiles['PROGRAM']=='GRAY'
    ii_dark_0 = (tiles['PROGRAM']=='DARK') & (tiles['PASS']==0)
    ii_dark_1 = (tiles['PROGRAM']=='DARK') & (tiles['PASS']==1)
    ii_dark_2 = (tiles['PROGRAM']=='DARK') & (tiles['PASS']==2)
    ii_dark_3 = (tiles['PROGRAM']=='DARK') & (tiles['PASS']==3)

    footprint = dict()
    footprint['gray'] = tiles[ii_gray]
    footprint['dark0'] = tiles[ii_dark_0]
    footprint['dark1'] = tiles[ii_dark_1]
    footprint['dark2'] = tiles[ii_dark_2]
    footprint['dark3'] = tiles[ii_dark_3]

    footprint['gray'].write('footprint/subset_gray.fits', overwrite=True)
    footprint['dark0'].write('footprint/subset_dark0.fits', overwrite=True)
    footprint['dark1'].write('footprint/subset_dark1.fits', overwrite=True)
    vstack([footprint['dark2'], footprint['dark3']]).write('footprint/subset_dark2_dark3.fits', overwrite=True)
    vstack([footprint['dark1'], footprint['dark2'], footprint['dark3']]).write('footprint/subset_dark1_dark2_dark3.fits', overwrite=True)
    vstack([footprint['dark0'], footprint['dark1'], footprint['dark2'], footprint['dark3']]).write('footprint/subset_dark0_dark1_dark2_dark3.fits', overwrite=True)
    vstack([footprint['gray'], footprint['dark0'], footprint['dark1'], footprint['dark2'], footprint['dark3']]).write('footprint/subset_gray_dark0_dark1_dark2_dark3.fits', overwrite=True)

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
    
def tile_efficiency(qa_json_file):
    f = open(qa_json_file)
    qa_dict = json.load(f)
    f.close()
    assign_total = []
    assign_science= []
    assign_sky = []
    assign_std = []
    for k in qa_dict:
        assign_total.append(qa_dict[k]['assign_total'])
        assign_science.append(qa_dict[k]['assign_science'])
        assign_sky.append(qa_dict[k]['assign_sky'])
        assign_std.append(qa_dict[k]['assign_std'])
    assign_total = np.array(assign_total)
    assign_science = np.array(assign_science)
    assign_sky = np.array(assign_sky)
    assign_std = np.array(assign_std)
    n_not_enough_sky = np.count_nonzero(assign_sky<400)
    n_not_enough_std = np.count_nonzero(assign_std<100)
    f_unassigned = (5000 - assign_total)/5000
    print(n_not_enough_sky, n_not_enough_std, np.median(f_unassigned))
    
def run_strategy(footprint_names, pass_names, strategy):
    for i_pass in range(4):
    
        footprint_name = footprint_names[i_pass]
        old_pass_name = pass_names[i_pass-1]
        pass_name = pass_names[i_pass]
        new_pass_name = pass_names[i_pass+1]
    
        os.makedirs('{}/fiberassign_{}'.format(strategy, pass_name), exist_ok=True)
    
        assign_footprint_filename = 'footprint/subset_{}.fits'.format(footprint_name)
        zcat_footprint_filename = 'footprint/subset_{}.fits'.format(pass_name)
        fiberassign_dir = '{}/fiberassign_{}/'.format(strategy, pass_name)
        mtl_filename = '{}/targets/{}_subset_dr8_mtl_dark_gray_northern_cap.fits'.format(strategy, pass_name)
        new_mtl_filename = '{}/targets/{}_subset_dr8_mtl_dark_gray_northern_cap.fits'.format(strategy, new_pass_name)
        old_zcat_filename = '{}/zcat/{}_zcat.fits'.format(strategy, old_pass_name)
        zcat_filename = '{}/zcat/{}_zcat.fits'.format(strategy, pass_name)
    
        if i_pass == 0:
            shutil.copyfile(initial_mtl_file, mtl_filename)
        
    
        # Run fiberassign
        cmd = 'fiberassign --mtl {} --sky targets/subset_dr8_sky.fits '.format(mtl_filename)
        cmd +=' --footprint {} --outdir {} --overwrite'.format(assign_footprint_filename, fiberassign_dir)
        print(cmd)
        os.system(cmd)
        #! $cmd
    
        # Gather fiberassign files
        fba_files = np.sort(glob.glob(os.path.join(fiberassign_dir,"tile*.fits")))

        # remove tilefiles that are not in the list of tiles to build zcat
        footprint = Table.read(zcat_footprint_filename)
        to_keep = []
        for i_file, fba_file in enumerate(fba_files):
            fibassign, header = fits.getdata(fba_file, header=True)
            tileid = header['TILEID'] 
            if tileid in footprint['TILEID']:
            #print(tileid, 'in list', zcat_footprint_filename)
            #print('keeping {}'.format(fba_file))
                to_keep.append(i_file)
            else:
                fiberassign_file = fba_file.replace('tile-', 'fiberassign_')
                if os.path.exists(fiberassign_file):
                    renamed_file = fiberassign_file.replace('.fits', '_unused.fits')
                    print(fiberassign_file, renamed_file)
                    os.rename(fiberassign_file, renamed_file)
            
        fba_files = fba_files[to_keep]
        print(len(fba_files))
            
        # Run qa
        cmd = "fba_run_qa --dir {} --footprint {}".format(fiberassign_dir, zcat_footprint_filename)
        print(cmd)
        os.system(cmd)
        #! $cmd
    
        # Read targets and truth
        targets = Table.read(mtl_filename)
        truth = Table.read(initial_truth_file)
    
        # Compute zcat
        if i_pass==0:
            zcat = desisim.quickcat.quickcat(fba_files, targets, truth, perfect=True)
        else:
            old_zcat = Table.read(old_zcat_filename)
            zcat = desisim.quickcat.quickcat(fba_files, targets, truth, zcat=old_zcat, perfect=True)        
    
        zcat.write(zcat_filename, overwrite=True)
        mtl = desitarget.mtl.make_mtl(targets, 'DARK|GRAY', zcat=zcat)
        mtl.write(new_mtl_filename, overwrite=True)


footprint_names = ['gray', 'dark0', 'dark1', 'dark2_dark3', 'full']
pass_names = ['gray', 'dark0', 'dark1', 'dark2_dark3', 'full']
run_strategy(footprint_names, pass_names, 'strategy_A')

footprint_names = ['gray_dark0_dark1_dark2_dark3', 'dark0_dark1_dark2_dark3', 'dark1_dark2_dark3', 'dark2_dark3', 'full']
pass_names = ['gray', 'dark0', 'dark1', 'dark2_dark3', 'full']
run_strategy(footprint_names, pass_names, 'strategy_B')

