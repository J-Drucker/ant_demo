#############################
# ANT demo
#############################

##### By Jonathan Drucker, 2025-06-20 #####

#####################################
# Dependencies
#####################################
import mne
from antio import read_cnt
from antio.parser import read_triggers
import pandas as pd
import numpy as np
import seaborn as sn
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os
import datetime
from io import StringIO


###############################
# Global variables
###############################
df_montage = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'parameters', 'WG_Net_NA-261_positions.csv'), index_col=0)
df_event_codes = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'parameters', 'event_code_definitions.csv'), dtype=str)


#####################################################################
#####################################################################

############################################
# Loading data from CNT
############################################

def load_CNT(data_dir, filename, rereference=True, set_channel_positions=True, adjust_annotations=True, crop_to_exp = True, preload=True):
        
    ##################################
    # Sub-function definitions
    ##################################
    # Re-reference to pseudo-mastoids
    def _rereference(raw):
        try:
            raw.add_reference_channels('5Z')
            raw.set_eeg_reference(['2LD', '2RD'])
            raw.drop_channels(['2LD', '2RD'])    
            print('Re-referenced to pseudo-mastoids')
        except:
            print('Re-referencing unsuccessful. Reference is still probably 5Z')

    # Load and set the channel positions
    def _set_channel_positions(raw):
        try:
            dm = mne.channels.make_dig_montage(
                ch_pos={x:df_montage.loc[x].values/1000 for x in df_montage.index if x in raw.ch_names},
                nasion=df_montage.loc['Nasion',:].values,
                lpa=df_montage.loc['LeftEar',:].values,
                rpa=df_montage.loc['RightEar',:].values,
            )
            raw.set_montage(dm)
            print('Channel positions loaded for WG_Net_NA-261')
        except:
            print('Channel position import for WG_Net_NA-261 unsuccessful')

    def _adjust_annotations(raw):
        try:
            # Rename the annotations
            event_codes = df_event_codes.set_index('code')['description'].to_dict()
            raw.annotations.rename(mapping={k:v for k,v in event_codes.items() if k in raw.annotations.description})
            print('Annotations (events) renamed')
        except:
            print('Annotations (events) renaming unsuccessful')
        else:
            try:
                # Redefine eyes open/closed to be single annotations with nonzero durations (instead of start/stop markers)
                a = mne.Annotations([],[],[],orig_time=raw.annotations.orig_time) # We will populate this empty Annotations object, and then add it into raw
                for i,x in enumerate(raw.annotations):
                    for cond in ['open', 'closed']:
                        if x['description'] == f"eyes_{cond}_start":
                            dur = min([y['onset']-x['onset'] for y in raw.annotations if (y['description']==f"eyes_{cond}_stop") and (y['onset'] > x['onset'])],
                            default=None) # Sometimes the fie ends abeuptly, e.g., if the paradigm wasn't completed
                            if dur is None: continue # That's fine; just skip this one
                            a.append(onset=x['onset'], duration=dur, description=f"eyes_{cond}", ch_names=None)
                raw.annotations.delete([i for i,x in enumerate(raw.annotations) if any([cond in x['description'] for cond in ['open_','closed_']])])
                raw.set_annotations(raw.annotations + a)
                print('Eyes open/closed recoded as single annotations with durations of ~20s')
            except Exception as e:
                print('Eyes open/closed recoding unsuccessful:\n',e)

    def _crop_to_exp(raw):
        kwargs = {}
        for annot in raw.annotations:
            if annot['description']=='exp_start':
                kwargs['tmin'] = annot['onset']
            if annot['description']=='exp_stop':
                kwargs['tmax'] = annot['onset']
        raw.crop(**kwargs)

    ##################################
    # Load and process data
    ##################################
    # Open the data as an mne.Raw object
    fullpath = os.path.join(data_dir, filename)
    print(f"Loading {fullpath}")
    raw = mne.io.read_raw_ant(fullpath, preload=preload)

    # Get the impedances using the older version of antio
    # These can be extracted with:
    #     df = pd.read_json(StringIO(raw.info['description']), orient='index').set_index(['time','channel'])
    #     or by using the get_impedances(raw) function defined in this file
    cnt = read_cnt(os.path.join(data_dir, filename))
    onsets,_,descriptions,impedances,_ = read_triggers(cnt)
    # Place them in the mne.Info object as a DataFrame
    raw.info['description'] = pd.DataFrame(
        impedances, # 2xN array where N is the number of channels. Impedances collected at the beginning and end of a recording
        index=pd.Index( # Timestamps for the impedance measures
            [onset/raw.info['sfreq'] for i,onset in enumerate(onsets) if descriptions[i]=='impedance'], # divide the sample number by the sample frequency to get the time in seconds
            name='time',
        ),
        columns=pd.Index( # Channel names
            raw.ch_names,
            name='channel',
        ),
    ).stack().rename('impedance').reset_index().to_json(orient='index') # Rearrange and convert to json string

    # Process data
    if rereference: _rereference(raw.load_data())
    if set_channel_positions: _set_channel_positions(raw)
    if adjust_annotations: _adjust_annotations(raw)
    if crop_to_exp: _crop_to_exp(raw)

    return raw

def get_impedances(raw):
    return pd.read_json(StringIO(raw.info['description']), orient='index').set_index(['time','channel'])


#####################################################
# Analyses
#####################################################
def analyze_full_psd(raw_unfilt, parent_dir, close_plot=False):
    # Filter for clarity
    raw = raw_unfilt.copy().filter(1,30).notch_filter(60)
    # Plot power spectra for the entire file
    fig, (ax1, ax2) = plt.subplots(2,1)
    raw_unfilt.plot_psd(fmax=80, show=True, ax=ax1)
    raw.plot_psd(fmax=80, show=True, ax=ax2)
    ax1.set(title='Raw unfiltered data')
    ax2.set(title='Filtered data (1-30Hz bandpass, 60Hz notch)')
    clean_up_plot(fig, os.path.join(parent_dir, 'output'), 'Power spectrum of full dataset.png', close_plot)
    return raw

def analyze_alpha(raw, parent_dir, fmax=35, close_plot=False):
    # Fourier transform
    psds = {cond: # separate PSD for eye open vs. closed
        mne.concatenate_raws( # stich the segments together
            raw.crop_by_annotations( # pull out segments of eyes open and eyes closed
                annotations=[a for a in raw.annotations if a['description'] == cond] # identify eyes open and eyes closed annotations
            )).compute_psd(method='welch', fmin=1, fmax=fmax, n_per_seg=int( # apply the welch method to compute psd (FFT in a sliding window)
            2**np.ceil(np.log2(raw.info['sfreq'])))) # Using a number of samples per window that's a power of 2, duration at least 1 second
        for cond in ['eyes_open','eyes_closed']}
    # Plots
    ## 1
    fig, (ax1, ax2) = plt.subplots(2,1)
    psds['eyes_open'].plot(axes=ax1)
    psds['eyes_closed'].plot(axes=ax2)
    ax1.set(title='Eyes Open')
    ax2.set(title='Eyes Closed')
    clean_up_plot(fig, os.path.join(parent_dir, 'output'), 'Power spectrum eyes open vs.closed with MNE.png', close_plot)
    ## 2
    bands = {'Delta (0-4 Hz)': (0, 4),'Theta (4-8 Hz)': (4, 8),'Alpha (8-12 Hz)': (8, 12), 'Beta (12-30 Hz)': (12, 30)}
    fig, axs = plt.subplots(2,4)
    psds['eyes_open'].plot_topomap(bands=bands, axes=axs[0,0:4], cmap='RdBu_r')
    psds['eyes_closed'].plot_topomap(bands=bands, axes=axs[1,0:4], cmap='RdBu_r')
    fig.suptitle('Frequency band maps: Eyes open (top) and Eyes closed (bottom)')
    clean_up_plot(fig, os.path.join(parent_dir, 'output'), 'Frequency mapping eyes open vs.closed with MNE.png', close_plot)
    ## 3-5
    df = pd.concat({k:v.to_data_frame(long_format=True, index='freq') for k,v in psds.items()})
    df = df.reset_index().drop(columns='ch_type').rename(columns={'value':'power', 'level_0':'condition'})
    df['power (log)'] = np.log(df['power'])
    fig = sn.relplot(df, kind='line', x='freq', y='power (log)', hue='condition')
    clean_up_plot(fig.figure, os.path.join(parent_dir, 'output'), 'Average power spectrum eyes open vs. closed.png', close_plot)
    fig = sn.relplot(df, kind='line', x='freq', y='power (log)', hue='condition', style='channel', alpha=.5)
    clean_up_plot(fig.figure, os.path.join(parent_dir, 'output'), 'Power spectrum eyes open vs. closed.png', close_plot)
    fig = sn.relplot(df, kind='line', x='freq', y='power (log)', hue='condition', col='channel', col_wrap=8)
    clean_up_plot(fig.figure, os.path.join(parent_dir, 'output'), 'Power spectrum eyes open vs. closed - per channel.png', close_plot)

def analyze_oddball(raw_unfilt, parent_dir, close_plot=False):
    raw = raw_unfilt.copy().filter(.1,30).notch_filter(60) # Aggressive, but reasonable, filter for ERPs
    epochs = mne.Epochs(raw, # Epoch by condition (frequent "O" vs. rare "X")
                        events = mne.events_from_annotations(raw, event_id={'frequent': 1,'rare': 2,})[0],
                        event_id = {'frequent':1, 'rare':2},
                        tmin = -0.2, tmax = .8, baseline = (-0.2, 0), # Look at the -200ms to 800ms window, baselining by the 200ms preceding the stimulus
                       )
    epochs.drop(epochs.to_data_frame(index=['time','condition','epoch']).groupby('epoch').std().max(axis=1).sort_values(ascending=False).index[0:int(.05*len(epochs))]) # Quick and dirty, drop the 5% noisiest epochs per group
    evoked = epochs.average(by_event_type=True) # Average across epochs within condition
    # Two plots: one without, one with the difference wave
    fig = mne.viz.plot_compare_evokeds(evoked, 
                                       picks=[ch for ch in evoked[0].ch_names if ((ch in [
                                           '5L','6L','7L','8L','9L',
                                           '5Z','6Z','7Z','8Z','9Z',
                                           '5R','6R','7R','8R','9R',
                                       ]))],
                                       show_sensors=True,
                                       combine='mean',
                                       title='P300',
                                      ) # plot the ERPs, avergaing across parietal channels
    clean_up_plot(fig[0], os.path.join(parent_dir, 'output'), 'P300_averaged.png', close_plot)
    # Compute the difference wave and add it to the mne.Evoked object (and then plot again)
    dw = mne.combine_evoked(evoked, [-1,1])
    dw.comment = 'difference_wave'
    evoked.append(dw)
    fig = mne.viz.plot_compare_evokeds(evoked, 
                                       picks=[ch for ch in evoked[0].ch_names if ((ch in [
                                           '5L','6L','7L','8L','9L',
                                           '5Z','6Z','7Z','8Z','9Z',
                                           '5R','6R','7R','8R','9R',
                                       ]))],
                                       show_sensors=True,
                                       combine='mean',
                                       title='P300 with difference wave',
                                       linestyles=['solid','solid','dashed'],
                                      ) # plot the ERPs, avergaing across parietal channels
    clean_up_plot(fig[0], os.path.join(parent_dir, 'output'), 'P300_averaged_with-difference-wave.png', close_plot)
    # Plot butterfly ERP + topo for "rare" only
    fig = evoked[1].plot_joint(title='Response to Rare Stimuli', times=[0,.100,.200,.300,.400,.500,.600]) 
    clean_up_plot(fig, os.path.join(parent_dir, 'output'), 'Response to Rare Stimuli.png', close_plot)
    return epochs,evoked
    

def analyze_independent_components(raw_unfilt, parent_dir, close_plot=False):
    ica = mne.preprocessing.ICA() # initialize the model
    raw = mne.concatenate_raws(raw_unfilt.copy().filter(2,30).resample(sfreq=128).crop_by_annotations(raw_unfilt.annotations[raw_unfilt.annotations.description=='eyes_open'])) # aggressive filter
    raw.info['bads'] = [ch for ch,noisy in zip(raw.ch_names, raw.get_data().std(axis=1) > .00008) if noisy] # ignore channels with a large SD
    raw_unfilt.info['bads'] = raw.info['bads']
    ica.fit(raw)

    fig_components = ica.plot_components(range(15))
    clean_up_plot(fig_components, os.path.join(parent_dir, 'output'), 'ICA topographies.png', close_plot)
    fig_sources = ica.plot_sources(raw, range(15))
    clean_up_plot(fig_sources, os.path.join(parent_dir, 'output'), 'ICA time series.png', close_plot)

    return ica

def apply_independent_components(raw_unfilt, parent_dir, ica, artifact_ics, close_plot=False):
    fig_overlay = ica.plot_overlay(raw_unfilt, exclude=artifact_ics)
    clean_up_plot(fig_overlay, os.path.join(parent_dir, 'output'), 'ICA overlay.png', close_plot)
    raw_ica = ica.apply(raw_unfilt.copy(), exclude=artifact_ics)
    return raw_ica



#####################################################
# Plotting
#####################################################
def clean_up_plot(fig, output_dir, name, close_plot=True):
    fig.set_size_inches(16,9)
    plt.get_current_fig_manager().window.showMaximized()
    fig.savefig(os.path.join(output_dir,name), dpi=300)
    if close_plot: plt.close(fig)