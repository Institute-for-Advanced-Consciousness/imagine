
import os
import sys
import numpy as np
import mne
import pyxdf

# ---------------- 1. Setting up environment and utility functions -----------------

def setup_environment():
    mne.set_log_level('WARNING')

setup_environment()

def load_data(base_directory, subject, session_number, session):
    filepath = os.path.join(base_directory, subject, str(session_number+1), f"{session}.xdf")
    data, _ = pyxdf.load_xdf(filepath)
    return data

def extract_eeg(data):
    eeg_stream = [stream for stream in data if stream["info"]["name"][0] == "eeg"][0]
    marker_stream = [stream for stream in data if stream["info"]["name"][0] == "markers"][0]

    eeg_data = np.array(eeg_stream["time_series"]).T
    eeg_ch_names = [ch["label"][0] for ch in eeg_stream["info"]["desc"][0]["channels"][0]["channel"]]
    sfreq = int(round(float(eeg_stream["info"]["nominal_srate"][0])))

    info = mne.create_info(ch_names=eeg_ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(eeg_data, info)

    eeg_time_stamps = eeg_stream['time_stamps']
    marker_time_stamps = marker_stream['time_stamps']

    overlap_start = max(eeg_time_stamps[0], marker_time_stamps[0])
    overlap_end = min(eeg_time_stamps[-1], marker_time_stamps[-1])

    eeg_data = eeg_data[:, (eeg_time_stamps >= overlap_start) & (eeg_time_stamps <= overlap_end)]
    eeg_time_stamps = eeg_time_stamps[(eeg_time_stamps >= overlap_start) & (eeg_time_stamps <= overlap_end)]
    raw = mne.io.RawArray(eeg_data, info)

    marker_data = []
    marker_times = []
    for j, timestamp in enumerate(marker_time_stamps):
        if overlap_start <= timestamp <= overlap_end:
            marker_data.append(marker_stream['time_series'][j])
            marker_times.append(timestamp)

    marker_data = np.array(marker_data)
    marker_times = np.array(marker_times) - overlap_start

    # Flatten the marker data
    marker_data_flat = marker_data.flatten()
    annotations = mne.Annotations(marker_times, np.repeat(0, len(marker_times)), marker_data_flat)
        
    raw.set_annotations(annotations)
    
    return raw

# ---------------- 3. Preprocessing steps -----------------

def plot_raw(raw):
    """
    Open an interactive plot to manually mark bad channels.
    
    Parameters:
    - raw: mne Raw object.
    
    Returns:
    - raw: mne Raw object with bad channels marked.
    """
    # Plot the raw data
    raw.plot(n_channels=64, scalings='auto', show=True, block=False)
    
    # Return the raw object with marked bad channels
    return raw

class SuppressPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        "to supress mne info output"
        sys.stdout.close()
        sys.stdout = self._original_stdout


def save_marker_data(data, subject, session_number, session, save_directory):
    # Extract the marker stream
    marker_stream = [stream for stream in data if stream["info"]["name"][0] == "markers"][0]
    
    # Convert marker data and time stamps to numpy arrays
    marker_data = np.array(marker_stream['time_series'])
    marker_time_stamps = np.array(marker_stream['time_stamps'])

    # Create a 2D array with time stamps in the first column and marker data in the second
    combined_data = np.column_stack((marker_time_stamps, marker_data))

    # Save the combined data
    np.savetxt(os.path.join(save_directory, f"{session}_marker_data.txt"), combined_data, fmt='%s')
    
def average_peak_to_peak_window(raw, start_sec, end_sec):
    start_sample = raw.time_as_index(start_sec)[0]
    end_sample = raw.time_as_index(end_sec)[0]
    
    data = raw.get_data(start=start_sample, stop=end_sample)  # shape is (n_channels, n_timepoints)
    peak_to_peak_per_channel = np.max(data, axis=1) - np.min(data, axis=1)
    average_peak_to_peak = np.mean(peak_to_peak_per_channel)
    
    # Get unit of data
    unit = raw.info['chs'][0]['unit']
    if unit == mne.io.constants.FIFF.FIFF_UNIT_V:
        scale_factor = raw.info['chs'][0]['cal']
        if scale_factor == 1e-6:
            unit_str = 'µV'
        elif scale_factor == 1e-3:
            unit_str = 'mV'
        else:
            unit_str = 'V'
    else:
        unit_str = 'unknown unit'

    return f"{average_peak_to_peak} {unit_str}"

import ipywidgets as widgets

def interactive_exclude_components(ica, raw_eeg_only):
    n_components = ica.get_components().shape[0]
    
    # Get the explained variance ratio for each ICA component
    explained_var_ratios = [ica.get_explained_variance_ratio(inst=raw_eeg_only, components=i, ch_type='eeg') for i in range(n_components)]
    
    # Create checkboxes with the explained variance percentage in their description
    checkboxes = [
        widgets.Checkbox(value=False, description=f'Component {i} - {evr["eeg"]*100:.2f}%') 
        for i, evr in enumerate(explained_var_ratios)
    ]
    
    # Display checkboxes
    checkbox_container = widgets.VBox(checkboxes)
    display(checkbox_container)
    
    # Update function to set excluded components
    def update_exclusions(b):
        selected_components = [i for i, cb in enumerate(checkboxes) if cb.value]
        ica.exclude += selected_components
        print(f"Excluded components: {selected_components}")
    
    # Button to finalize selections
    update_button = widgets.Button(description="Update Exclusions")
    update_button.on_click(update_exclusions)
    display(update_button)

import ipywidgets as widgets
from IPython.display import display

def add_bad_segments(raw_eeg_only):
    print("Enter bad segments in the format 'start-stop' separated by commas (e.g., '100-150, 200-250'). Press the button when done.")

    bad_segments = []

    def on_submit(change):
        segment_input = text.value.strip()

        segments = segment_input.split(',')
        valid_segments = []
        has_invalid = False
        for segment in segments:
            if '-' not in segment:
                print(f"Invalid segment '{segment.strip()}'. Please enter in the format 'start-stop'.")
                has_invalid = True
                break
            try:
                start, stop = map(float, segment.split('-'))
                valid_segments.append((start, stop))
            except ValueError:
                print(f"Invalid segment '{segment.strip()}'. Please enter numeric values for start and stop.")
                has_invalid = True
                break

        if not has_invalid:
            bad_segments.extend(valid_segments)
            # Adding the bad segments as annotations to the raw object
            for start, stop in valid_segments:
                raw_eeg_only.annotations.append(start, stop - start, 'BAD_manual')
            print("Segments added!")
            text.value = ''  # Clear the input field for a new entry

    text = widgets.Text(value='', placeholder='Type segments here...', description='Segments:', disabled=False)
    button = widgets.Button(description="Add Segments")
    button.on_click(on_submit)
    
    display(text, button)

import os

def get_sessions(subject):
    cwd = os.getcwd()
    base_directory = os.path.join(cwd, '..', '..', 'raw_data')
    subject_directory = os.path.join(base_directory, subject)

    session_1_directory = os.path.join(subject_directory, '1')
    session_2_directory = os.path.join(subject_directory, '2')

    session_1 = None
    session_2 = None

    if os.path.exists(session_1_directory):
        session_1_files = os.listdir(session_1_directory)
        if session_1_files:
            session_1 = session_1_files[0].replace('.xdf', '')
        else:
            raise FileNotFoundError(f"No files found in directory {session_1_directory}")
    else:
        raise FileNotFoundError(f"Directory {session_1_directory} does not exist")

    if os.path.exists(session_2_directory):
        session_2_files = os.listdir(session_2_directory)
        if session_2_files:
            session_2 = session_2_files[0].replace('.xdf', '')
        else:
            raise FileNotFoundError(f"No files found in directory {session_2_directory}")
    else:
        raise FileNotFoundError(f"Directory {session_2_directory} does not exist")

    return session_1, session_2
