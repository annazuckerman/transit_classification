'''
Functions for use in Transit Classification aglorithms
'''

##### Data loading and pre-processing functions #####

import numpy as np
import pandas as pd
import lightkurve as lk
from scipy.optimize import curve_fit
import csv
import os

def linear_func(x, a, b):
    """
    A simple linear trend for detrending.
    ----------
    Parameters:
        x (numpy array): x-values
        a (float): slope
        b (float): intercept
    ----------
    Returns:
        y (numpy array): linear function of x
    """
    return a*x + b

def stitch(lc_collection): 
    """
    Linearly detrend each "quarter" (interval of observing) of the data, 
    and combine them together. We do this to correct for offsets between quarters 
    and linear systmatics within each quarter. These are instrumental effects.
    ----------
    Parameters:
        lc_collection (lightcurve collection): the collection of quarters of lightcurve
                data returned by the Lightkurve query.
    ----------
    Returns:
        tot_time (numpy array): stitched time values
        tot_flux (numpy array): stitched and detrended flux values
        tot_flux_err (numpy array): stitched flux error values
        tot_qual (numpy array): stitched data quality flag values
    """
    tot_time = np.zeros(0) # to store concatenated arrays
    tot_flux = np.zeros(0)
    tot_flux_err = np.zeros(0)
    tot_qual = np.zeros(0)
    for i in range(len(lc_collection)):
        lc = lc_collection[i]
        flux = lc.flux.value
        time = lc.time.value
        flux_err = lc.flux_err.value
        qual = lc.quality
        rel_flux_err = flux_err/flux
        nan_mask = np.invert(np.isnan(flux))
        
        # Fit and remove linear trend
        popt, pcov = curve_fit(linear_func, time[nan_mask], flux[nan_mask])
        linear_trend = linear_func(time, *popt) # evaluate over the whole interval
        norm = flux / linear_trend
        
        tot_time = np.hstack([tot_time, time])
        tot_flux = np.hstack([tot_flux, norm])
        tot_flux_err = np.hstack([tot_flux_err, rel_flux_err])
        tot_qual = np.hstack([tot_qual, qual])
        
    return tot_time, tot_flux, tot_flux_err, tot_qual

def collect_curves(n_curves, n_timesteps=1000, downsize_method='interpolate', pct_transit=48.8):
    """
    Collect raw light curves into flux and time arrays (of shape [n_curves, n_timesteps]).
    Construct corresponding array (of shape [n_curves]) containing labels (1 = transit, 0 = no transit).
    NOTE: downloads take a long time, so I recomend saving the resulting arrays as csv files.
    ----------
    Parameters:
        n_curves: number of curves to download (will be approximate if percentages don't work out)
        n_timesteps: number of timesteps to interpolate to (at some point we could also try truncation)
        downsize_method: method to force curves to the n_timesteps. Options are 'interpolate' or 'truncate'.
        pct_transit: percent of returned dataset that contains a transt. Default is 48.8, which is 
                    the overall perentage of the 150,000 available Keplar curves that have transits. 
    ----------
    Returns:
        all_curves: numpy array of shape [n_curves, n_timesteps] containing light curve flux values
        all_times: numpy array of shape [n_curves, n_timesteps] containing light curve time values
        all_labels: 1-dimensional array containing correponding transit labels (1 = transit, 0 = no transit)
    """

    # Get IDs of non-transit curves
    data = pd.read_csv('Data/exoplanet_archive_KOIs.csv')
    all_nontransit_ids = data.loc[data['koi_disposition'] == 'FALSE POSITIVE']['kepid'].to_list()
    nontransit_ids = np.random.choice(all_nontransit_ids, size = int(n_curves*(1-pct_transit/100)))
    all_ids = np.copy(nontransit_ids) 

    # Get IDs of transit curves 
    all_transit_ids = data.loc[data['koi_disposition'] == 'CONFIRMED']['kepid'].to_list()
    transit_ids = np.random.choice(all_transit_ids, size = int(n_curves*(pct_transit/100)))
    all_ids = np.concatenate((all_ids, transit_ids))

    # Randomize id list 
    all_ids = all_ids[np.random.permutation(len(all_ids))]

    # Fill array with transit and non-transit curves
    all_curves = np.zeros((len(all_ids), n_timesteps))
    all_times = np.zeros((len(all_ids), n_timesteps))
    all_labels = np.zeros(len(all_ids))
    i = 0
    print(f'Downloading {len(all_ids)} light curves')
    for star in all_ids:
        star = int(star)
        print(f'\tStar {i}', end='\r')
        # Download full light curve
        curve = lk.search_lightcurve(f'KIC{star}', author='Kepler', cadence='long').download_all()
        # "Stich" together quarters
        time, flux, flux_err, quality = stitch(curve)
        # Set poor quality data to NaN
        good = (quality == 0) * (flux_err > 0) * (np.isfinite(time)) * (np.isfinite(flux)) * (np.isfinite(flux_err))
        flux[np.invert(good)] = np.NaN
        # Force to length n_timesteps
        if downsize_method == 'interpolate':
            flux = np.interp(np.linspace(0,n_timesteps-1,n_timesteps), np.linspace(0,n_timesteps-1,len(flux)), flux)
            time = np.interp(np.linspace(0,n_timesteps-1,n_timesteps), np.linspace(0,n_timesteps-1,len(time)), time)
        if downsize_method == 'truncate':
            flux = flux[0:n_timesteps]
            time = time[0:n_timesteps]
        # Get label
        label = int(1) if id in transit_ids else int(0)
        # Add time and flux to arrays
        all_times[i] = time
        all_curves[i] = flux
        all_labels[i] = label
        i += 1

    return all_curves, all_times, all_labels

def collect_curves_tofiles(n_curves, n_timesteps=1000, downsize_method='interpolate', phase_fold = False, smooth = False, pct_transit=48.8, savepath='../LC_Data'):
    """
    Add raw light curves (row-wise) into csv files storing flux, time, and labels (1 = transit, 0 = no transit).
    Every call to this function will add rows to these csv files. 
    NOTE: this should save the file if the error is lightcurve failing to download the next star, 
          but if something happens (e.g. user interupts) during file opening, I think sometimes the 
          files done't get saved correctly. 
    ----------
    Parameters:
        n_curves: number of curves to download (will be approximate if percentages don't work out)
        n_timesteps: number of timesteps to interpolate or truncate to. If "phase fold", number of bins to
                     use to bin folded lightcurve.
        downsize_method: method to force curves to n_timesteps. Options are 'interpolate' or 'truncate'.
        smooth: if True, will smooth returned lightcurves, if False will not.
        phase_fold: if True, will phase fold returned lightcurves, if False will not.
        pct_transit: percent of returned dataset that contains a transt. Default is 48.8, which is 
                    the overall perentage of the 150,000 available Keplar curves that have transits. 
        savepath = path in which to create the stored files
    ----------
    Generates or adds to the following 3 files:
        savepath/flux_all_[n_timesteps]_[pct_transits].csv: flux values, with each row representing one curve.
        savepath/time_all_[n_timesteps]_[pct_transits].csv: corresponding time values
        savepath/labels_all_[n_timesteps]_[pct_transits].csv: corresponding labels (1 per row)
    """

    # Get IDs of non-transit curves
    data = pd.read_csv('Data/exoplanet_archive_KOIs.csv')
    all_nontransit_ids = data.loc[data['koi_disposition'] == 'FALSE POSITIVE']['kepid'].to_list()
    nontransit_ids = np.random.choice(all_nontransit_ids, size = int(n_curves*(1-pct_transit/100)))
    all_ids = np.copy(nontransit_ids)

    # Get IDs of transit curves 
    all_transit_ids = data.loc[data['koi_disposition'] == 'CONFIRMED']['kepid'].to_list()
    transit_ids = np.random.choice(all_transit_ids, size = int(n_curves*(pct_transit/100)))
    all_ids = np.concatenate((all_ids, transit_ids))

    # Randomize id list 
    all_ids = all_ids[np.random.permutation(len(all_ids))]
 
    # Create files if they don't exist, else check that they have the same length
    filepaths = []
    filelengths = []
    for tag in ['flux', 'time', 'labels']:
        filepath = f"{savepath}/{tag}_all_{n_timesteps}_{pct_transit}.csv"
        filepaths.append(filepath)
        if os.path.exists(filepath) == False:
            print(f'Creating {filepath} to store {tag}')
            open(filepath, 'w')
            filelengths.append([0])
        else:
            print(f'Adding {tag} to {filepath}')
            filelengths.append(len(pd.read_csv(filepath, header=None, delimiter=',')))
    if all(element == filelengths[0] for element in filelengths) == False:
        raise Exception(f'{filepaths[0]}, {filepaths[1]}, and {filepaths[2]}, have different number of rows ({filelengths[0]},{filelengths[1]}, and {filelengths[2]}).')

    # Download curves and append to files
    with open(rf'{filepaths[0]}','a') as f1, open(rf'{filepaths[1]}','a') as f2, open(rf'{filepaths[2]}','a') as f3:
        writer1=csv.writer(f1)
        writer2=csv.writer(f2)
        writer3=csv.writer(f3)
        i = 0
        print(f'Downloading {len(all_ids)} light curves')
        for star in all_ids:
            star = int(star)
            # Get label
            label = 1 if (star in transit_ids) else 0
            print(f'\tStar {i}', end='\r')
            # Download full light curve
            curve = lk.search_lightcurve(f'KIC{star}', author='Kepler', cadence='long').download_all()
            # "Stich" together quarters
            time, flux, flux_err, quality = stitch_quarters(curve)
            if smooth:
                # Smooth to remove stellar variability 
                flux = flux/medfilt(flux,51)
            if phase_fold:
                # Phase fold
                curve = curve.stitch().flatten(window_length=901).remove_outliers()
                period = np.linspace(1, 30, 10000) # periods to search
                # Create a BLSPeriodogram
                bls = curve.to_periodogram(method='bls', period=period, frequency_factor=500)
                period = bls.period_at_max_power
                t0 = bls.transit_time_at_max_power
                dur = bls.duration_at_max_power
                folded = curve.fold(period=period, epoch_time=t0)
                folded_flux = folded.flux.value
                folded_time = folded.time.value
                planet_model = bls.get_transit_model(period=period,
                                       transit_time=t0,
                                       duration=dur).fold(period, t0)
                binned, bin_edges, binnumber = binned_statistic(folded_time, folded_flux, 'median', bins = n_timesteps)
                folded_smoothed_time = bin_edges[1:]
                folded_smoothed_flux = medfilt(binned,3)
                writer1.writerow(np.array(folded_smoothed_flux))
                writer2.writerow(np.array(folded_smoothed_time))
                writer3.writerow(np.array([label]))
                continue
            # Set poor quality data to NaN
            good = (quality == 0) * (flux_err > 0) * (np.isfinite(time)) * (np.isfinite(flux)) * (np.isfinite(flux_err))
            flux[np.invert(good)] = np.NaN
            # Force to length n_timesteps
            if downsize_method == 'interpolate':
                flux = np.interp(np.linspace(0,n_timesteps-1,n_timesteps), np.linspace(0,n_timesteps-1,len(flux)), flux)
                time = np.interp(np.linspace(0,n_timesteps-1,n_timesteps), np.linspace(0,n_timesteps-1,len(time)), time)
            if downsize_method == 'truncate':
                flux = flux[0:n_timesteps]
                time = time[0:n_timesteps]
            # Add to csv files
            writer1.writerow(np.array(flux))
            writer2.writerow(np.array(time))
            writer3.writerow(np.array([label]))
            i += 1
