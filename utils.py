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
from scipy.signal import medfilt
from scipy.stats import binned_statistic

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

def stitch_quarters(lc_collection): 
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
        time, flux, flux_err, quality = stitch_quarters(curve)
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


def collect_curves_tofiles(n_curves, n_timesteps=1000, downsize_method='interpolate', phase_fold=False, smooth=False, pct_transit=48.8, savepath='../LC_Data'):
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
                i = i+1
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

#### Functions for use with NNs ####
import torch
from torch import nn
from sklearn.datasets import make_classification
from torch.utils.data import Dataset, TensorDataset, DataLoader

def check_inputs(train_ds, train_loader):
    '''
    Succinctly check that dataloaders are constructed as they should be
    ''' 

    print('Train data:')
    print(f'     {len(train_ds)} obs, broken into {len(train_loader)} batches')
    train_features, train_labels = next(iter(train_loader))
    shape = train_features.size()
    print(f'     Each batch has data of shape {train_features.size()}, e.g. {shape[0]} obs, {shape[2]} pixels, {shape[1]} feature channels') #{[shape[2], shape[3]]} pixels each, {shape[1]} layers (features)')
    shape = train_labels.size()
    print(f'     Each batch has labels of shape {train_labels.size()}, e.g. {shape[0]} obs, {shape[1]} classes') #{[shape[2], shape[3]]} pixels each, {shape[1]} layers (classes)')


class MyDataset(Dataset):
  '''
  Pytorch Datasaet object to load in data
  '''

  def __init__(self, X1, y, X2=None, norm=True, impute_nans=True):
    self.x1 = X1
    self.x2 = X2
    self.y = y
    self.len = self.y.shape[0]
    self.norm = norm
    self.impute_nans = impute_nans
  
  def __getitem__(self, index):
    # Get lightcurve
    x = torch.from_numpy(self.x1.astype(np.float32))
    x = x[index]
    x_prenorm = x.clone().detach()
    if self.norm: 
      x = ((x - np.nanmin(x))/(np.nanmax(x) - np.nanmin(x))) # min-max normalization to [0, 1] 
    x_preimpute = x.clone().detach()
    if self.impute_nans: 
      x = torch.tensor(pd.Series(x).interpolate(limit_direction='both')) # fill NaNs with means of nieghboring values
      if torch.any(torch.isnan(x)): 
        print(f'{index} Initial {x_prenorm[0:100]}') 
        print(f'{index} After norm {x_preimpute[0:100]}')
        print(f'{index} SOME STILL NAN After impute {x[0:100]}\n')
        a=b
        # print(f'Are all befores nan?: {torch.any(torch.isnan(x_pre))}')
    if self.x2 is None:
        X = np.zeros((1, len(x)), dtype=np.float32) # Add dummy axis since only one channel
        X[0, :] = x 
    else: 
        t = torch.from_numpy(self.x2.astype(np.float32))
        t = t[index]
        X = np.zeros((2, len(x)), dtype=np.float32) # Add space for second channel
        X[0, :] = x 
        X[1, :] = t
    # Get label
    y = np.squeeze(torch.from_numpy(self.y))#.type(torch.LongTensor))
    y = self.y[index]
    mask = np.zeros(2, dtype=np.float32)
    mask[int(y)] = 1 # e.g. if y=0, first idx is 1, if y=1, second idx is 1
    return X, mask
    
  def __len__(self):
    return self.len


def validate(val_loader, model):
    '''
    Function to calculate validation accuracy
    '''
    TP, TN, FP, FN = 0, 0, 0, 0
    P = 0
    num_obs = 0
    model.eval() # set model into eval mode
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to("cpu")
            y = y.to("cpu")#.unsqueeze(1)
            probs = torch.sigmoid(model(x)) # call model to get output, apply sigmoid to get class probs [batch_size, n_classes]
            preds = np.argmax(probs, axis=1) # turn class probs into 1D predicted class [batch_size] by returning idx of max prob
            y = np.argmax(y, axis=1) # turn one-hot labels into 1D labels by returning idx where y is 1
            TP += len(np.where((preds == 1) & (y == 1))[0])
            TN += len(np.where((preds == 0) & (y == 0))[0])
            FP += len(np.where((preds == 1) & (y == 0))[0])
            FN += len(np.where((preds == 0) & (y == 1))[0])
            P += len(np.where(y == 1)[0])
            num_obs += x.shape[0]
    accuracy = (TP+TN)/num_obs*100
    # print(TP)
    # print(P)
    recall = TP/P*100

    return accuracy, recall


class LinearNN(nn.Module):
  '''
  Simple feed-foward linear NN
    - get rid of extra channels dim in last layer (channels not used in linear, but extra dim for completness)
  '''

  def __init__(self, input_dim, output_dim, n_feats): 
    super(LinearNN, self).__init__()
    self.layers = nn.ModuleList()
    n_feats = [input_dim] + n_feats + [output_dim]
    for i in range(len(n_feats)-1):
       self.layers.append(nn.Linear(n_feats[i], n_feats[i+1]))
    
  def forward(self, x):
    for i in range(len(self.layers)):
        x = self.layers[i](x) # torch.sigmoid(self.layers[i](x))
    x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
    return x


class ConvNN1(nn.Module):
  '''
  1D convolutional NN
    - based on https://www.kaggle.com/code/purplejester/pytorch-deep-time-series-classification
  '''

  def __init__(self, in_channels, out_classes, channels=[64, 128, 256], in_samps=1000, k_size=3): 
    super(ConvNN1, self).__init__()

    self.convs = nn.Sequential(
        nn.Conv1d(in_channels, channels[0], kernel_size=k_size, padding='same'), # [5, 64, 1000]
        nn.Conv1d(channels[0], channels[1], kernel_size=k_size, padding='same'), # [5, 128, 1000]
        nn.Conv1d(channels[1], channels[2], kernel_size=k_size, padding='same')) # [5, 256, 1000]
    
    self.linears = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(channels[-1]*in_samps, 5000),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(5000, 100), 
        nn.ReLU(inplace=True),
        nn.Linear(100, out_classes))

  def forward(self, x):
    x = self.convs(x) 
    x = x.view(x.size(0), -1)
    x = self.linears(x) 
    print(x.shape)
    return x


class ConvNN2(nn.Module):
  '''
  1D convolutional NN
  - Based on https://jovian.com/ningboming/time-series-classification-cnn
  '''

  def __init__(self, in_channels, out_classes, channels=[64, 128, 256], in_samps=1000, samps=[1024, 512], k_size=3): 
    super(ConvNN2, self).__init__()

    self.convs = nn.Sequential(
        nn.Conv1d(in_channels, channels[0], kernel_size=k_size, stride=1, padding=1), # [5, 64, 1000]
        nn.ReLU(),
        nn.Conv1d(channels[0], channels[1], kernel_size=k_size, stride=1, padding=1), # [5, 128, 1000]
        nn.ReLU(),
        nn.MaxPool1d(2),                                                              # [5, 128, 500]
        nn.Conv1d(channels[1], channels[2], kernel_size=k_size, stride=1, padding=1), # [5, 256, 500]
        nn.ReLU(),
        nn.Conv1d(channels[2], channels[2], kernel_size=k_size, stride=1, padding=1), # [5, 256, 500]
        nn.ReLU(),
        nn.MaxPool1d(2))                                                              # [5, 256, 250]

    self.linears = nn.Sequential(
        nn.Flatten(), 
        nn.Linear(int(channels[-1]*in_samps/4), samps[0]), # int(channels[-1]*in_samps/4)  pool by 2, x2, so samps dim / 4
        nn.ReLU(),
        nn.Linear(samps[0], samps[1]),
        nn.ReLU(),
        nn.Linear(samps[1], out_classes))

  def forward(self, x):
    x = self.convs(x) 
    x = self.linears(x) 
    return x  


class ConvNN3(nn.Module):
  '''
  Replication of ConvNN2 using modulists to make more flexible
  Also add dropouts
  '''

  def __init__(self, in_channels, out_classes, channels=[64, 128, 256], in_samps=500, samps=[1000, 200], k_size=3): 
    super(ConvNN3, self).__init__()

    
    channels = [in_channels] + channels + [channels[-1]]
    self.convs = nn.ModuleList()
    for i in range(len(channels)-1):
       self.convs.append(nn.Conv1d(channels[i], channels[i+1], kernel_size=k_size, stride=1, padding=1))
       self.convs.append(nn.ReLU())
    self.convs.insert(int(len(self.convs)/2), nn.MaxPool1d(2)) # insert maxpool halfway 
    self.convs.append(nn.MaxPool1d(2))

    samps = [int(channels[-1]*in_samps/4)] + samps
    self.linears = nn.ModuleList()
    self.linears.append(nn.Flatten())
    for i in range(len(samps)-1):
       self.linears.append(nn.Linear(samps[i], samps[i+1]))
       self.linears.append(nn.ReLU())
       self.linears.append(nn.Dropout(p=0.2))
    self.linears.append(nn.Linear(samps[-1], out_classes))

  def forward(self, x):
    for conv in self.convs:
        x = conv(x)
    for linear in self.linears:
        x = linear(x)
    return x  

class LSTM(nn.Module):
  '''
  Uses PyTorch built-in LSTM module
  '''

  def __init__(self, in_channels, out_classes, channels=[60], in_samps=500, samps=[1000, 200]): 
    super(LSTM, self).__init__()
    self.lstm = nn.LSTM(input_size=in_channels, hidden_size=channels[0], num_layers=3, batch_first=True) 
    self.linears = nn.Sequential(
            nn.Flatten(),                                # [5, 30000]
            nn.Linear(channels[-1]*in_samps, samps[0]),
            nn.Linear(samps[0], samps[1]),
            nn.Linear(samps[1], out_classes))

  def forward(self, x):
    x = torch.permute(x, (0,2,1)) # flip channels and samples dimensions to use nn.LSTM
    x, _ = self.lstm(x)
    x = torch.permute(x, (0,2,1)) # flip back
    x = self.linears(x)
    return x  