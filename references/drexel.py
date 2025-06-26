# Functions etc. for the Drexel analysis. Nothing fancy here.
#############################################################

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

##############################################################################################

####################################
# Global variables
####################################
# File paths that may differ for different users
parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
data_dir = os.path.join(parent_dir, 'data','EEG')
output_dir = os.path.join(parent_dir, 'output')

# Information about electrodes, event codes, etc
df_electrodes = pd.read_excel(os.path.join(parent_dir,'electrodes.xlsx'))[['channel','type','AP','Hemisphere']]

#########################
# Plotting functions
#########################

def plot_touchups_and_save(g, name=None):
    g.figure.set_size_inches(16,8)
    g.set(yscale='log')
    g.tight_layout()
    for ax in g.axes_dict.values():
        ax.yaxis.set_major_formatter(ScalarFormatter())
    if not name is None:
        g.savefig(os.path.join(output_dir,name), dpi=300)
        plt.close(g.figure)


############################################
# Loading data from CNT
############################################

def load_CNT(full_path, rereference=True, set_channel_positions=True, adjust_annotations=True):
    ##################################
    # Load and process data
    ##################################
    def _main_(full_path):
        # Open the data as an mne.Raw object
        raw = mne.io.read_raw_ant(full_path, preload=True)
        # Grab the sample rate
        sfreq = raw.info['sfreq']
    
        # Process data
        if rereference: _rereference(raw)
        if set_channel_positions: _set_channel_positions(raw)
        if adjust_annotations: _adjust_annotations(raw)
    
        return raw
        
    ##################################
    # Function definitions
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
            df_montage = pd.read_csv(os.path.join(parent_dir, 'WG_Net_NA-261_positions.csv'), index_col=0)
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
            df_event_codes = pd.read_csv(os.path.join(parent_dir, 'event_code_definitions.csv'), dtype=str)
            event_codes = df_event_codes.set_index('code')['description'].to_dict()
            raw.annotations.rename(mapping={k:v for k,v in event_codes.items() if k in raw.annotations.description})
            print('Annotations (events) renamed')
        except:
            print('Annotations (events) renaming unsuccessful')
        else:
            try:
                # Redefine eyes open/closed to be single annotations with nonzero durations (instead of start/stop markers)
                a = mne.Annotations([],[],[],orig_time=raw.annotations.orig_time) # Populate this empty Annotations object, and then add it into raw
                for i,x in enumerate(raw.annotations):
                    for cond in ['open', 'closed']:
                        if x['description'] == f"eyes_{cond}_start":
                            dur = min([y['onset']-x['onset'] for y in raw.annotations if (y['description']==f"eyes_{cond}_stop") and (y['onset'] > x['onset'])])
                            a.append(onset=x['onset'], duration=dur, description=f"eyes_{cond}", ch_names=None)
                raw.annotations.delete([i for i,x in enumerate(raw.annotations) if any([cond in x['description'] for cond in ['open_','closed_']])])
                raw.set_annotations(raw.annotations + a)
                print('Eyes open/closed recoded as single annotations with durations of ~20s')
            except:
                print('Eyes open/closed recoding unsuccessful')
    
    return _main_(full_path)

    
    
###########################
# Analyses
###########################

##### Imedpance analysis #####
def impedance_analysis(filename):
    # Open the data as an mne.Raw object
    raw = mne.io.read_raw_ant(os.path.join(data_dir, filename))
    # This grabs the impedances. Bit redundant. They'll fix that later I'm sure
    cnt = read_cnt(os.path.join(data_dir, filename))
    onsets,durations,descriptions,impedances,disconnect = read_triggers(cnt)
    print(descriptions)
    recording_start = cnt.get_start_time()
    sfreq = cnt.get_sample_frequency()
    # ONSETS is the time in samples that the impedance measurement occurred. Convert it to absolute time
    times = [recording_start + datetime.timedelta(seconds=onset/sfreq) for i,onset in enumerate(onsets) if descriptions[i]=='impedance']
    # Make a nice litte DataFrame and return it
    return pd.DataFrame(impedances, index=pd.Index(times, name='time'), columns=pd.Index(raw.ch_names, name='channel')).T

def sixty_hz_analysis(filename):
    # Open the data as an mne.Raw object
    raw = mne.io.read_raw_ant(os.path.join(data_dir, filename), preload=True)
    sfreq = raw.info['sfreq']
    if raw.duration < 10: return pd.DataFrame([])
    raw.add_reference_channels('5Z')
    raw.set_eeg_reference(['2LD', '2RD'])
    raw.drop_channels(['2LD', '2RD'])    
    print('Re-referenced to pseudo-mastoids')
    
    df_montage = pd.read_csv(os.path.join(parent_dir, 'WG_Net_NA-261_positions.csv'), index_col=0)
    dm = mne.channels.make_dig_montage(
        ch_pos={x:df_montage.loc[x].values/1000 for x in df_montage.index if x in raw.ch_names},
        nasion=df_montage.loc['Nasion',:].values,
        lpa=df_montage.loc['LeftEar',:].values,
        rpa=df_montage.loc['RightEar',:].values,
    )
    
    raw.set_montage(dm)
    print('Channel positions loaded')

    # Calculate the 60 Hz amplitude in a sliding window
    # Twenty-second epochs?
    epochs = mne.make_fixed_length_epochs(raw, duration=60, overlap=0)
    psd = epochs.compute_psd(method='welch', fmin=60, fmax=60)
    times = [raw.info['meas_date'] + datetime.timedelta(seconds=x) for x in psd.events[:,0]/sfreq]

    df = psd.to_data_frame(index=['condition','epoch','freq'])
    df.index = pd.Index(times, name='time')
    df = df.melt(var_name='channel', value_name='60Hz_uV', ignore_index=False).reset_index()
    df['60Hz_uV'] = df['60Hz_uV']**.5*1000000

    # Merge in the electrode info and put it in the list
    return pd.merge(df, df_electrodes, left_on='channel', right_on='channel')
