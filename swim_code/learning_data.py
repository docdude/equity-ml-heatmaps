import pandas as pd
import numpy as np
import utils
import os
import segmentation
import constants
import scipy.signal
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

class LearningData(object):

    def __init__(self):
        self.users = None
        self.labels = None
        self.labels_index = None
        self.stroke_labels = None
        self.data_dict = None
        self.data_windows = None
        self.data_path = None
        self.windows = None
        self.window_locs = None
        self.win_len = None
        self.slide_len = None
        self.columns = None
        self.data_columns = None
        self.data_columns_ix = None
        self.sensors = None
        self.mirror_prob = None
        self.mirror_ix = None
        self.noise_std = None
        self.time_scale_factors = None
        self.stroke_range = None
        self.group_probs = {'original': 1.0}
        self.label_type = None
        self.multi_class_thresh = None
        self.majority_thresh = None
        self.norm_type = None
        self.sensor_shift = None
        self.sensor_scale = None
        self.sensor_center = None 

    def normalize_recordings(self, detrend=None, norm_range=None, norm_type=None):
        """
        Normalize data on a recording-by-recording basis. Currently provides detrending and range normalization as
        possibly required by the pressure and light sensors.
        :param detrend: A dictionary of booleans whose keys are sensors. Specifies which sensors are detrended
        :param norm_range: A dictionary of booleans whose keys are sensors. Specifies which sensors are normalized
                           across the range.
        :return: self.data_dict is modified appropriately.
        """
        if detrend is None:
            detrend = {sensor: False for sensor in self.sensors}
        if norm_range is None:
            norm_range = {sensor: False for sensor in self.sensors}
            norm_range['LIGHT'] = True
        for user in self.users:
            print("Normalizing recordings for: %s" % user)
            for rec in self.data_dict['original'][user].keys():
                for sensor in self.sensors:
                    sensor_cols = [col for col in self.data_columns if col.startswith(sensor)]
                    if detrend[sensor] is True:
                        for col in sensor_cols:
                            self.data_dict['original'][user][rec][col] = \
                                utils.detrend(self.data_dict['original'][user][rec][col].values)
                    if norm_range[sensor] is True and norm_type == 'statistical':
                        for col in sensor_cols:
                            self.data_dict['original'][user][rec][col] = \
                                utils.normalize_zscore(self.data_dict['original'][user][rec][col].values)
                        #print(f"Finished applying zscore normalization {sensor_cols}")

                    if norm_range[sensor] is True and norm_type == 'range':
                        for col in sensor_cols:
                            self.data_dict['original'][user][rec][col] = \
                                utils.normalize_range(self.data_dict['original'][user][rec][col].values)
                        #print(f"Finished applying range(0-1) normalization {sensor_cols}")



    def normalize_global(self, norm_range=None):
        """
        Normalize data globally. Range normalization is the only feature available.
        :param norm: A dictionary of booleans whose keys are sensors. Specifies which sensors are normalized over the
                     range.
        :return: self.data_dict is modified appropriately
        """
        if norm_range is None:
            norm = {sensor: True for sensor in self.sensors}
        sensor_max = {key: -np.inf for key in self.sensors}
        sensor_min = {key: np.inf for key in self.sensors}
        print("Computing normalization values...")
        for user in self.data_dict['original'].keys():
            for rec in self.data_dict['original'][user].keys():
                for sensor in self.sensors:
                    if norm_range[sensor]:
                        cols = [col for col in self.data_columns if col.startswith(sensor)]
                        sensor_local_max = np.max(self.data_dict['original'][user][rec][cols].values)
                        sensor_local_min = np.min(self.data_dict['original'][user][rec][cols].values)
                        sensor_max[sensor] = np.max([sensor_max[sensor], sensor_local_max])
                        sensor_min[sensor] = np.min([sensor_min[sensor], sensor_local_min])
        sensor_shift = {key: sensor_min[key] for key in self.sensors}
        sensor_scale = {key: sensor_max[key] - sensor_min[key] for key in self.sensors}
        print("sensor_shift:", sensor_shift)
        print("sensor_scale:", sensor_scale)
        print("Finished computing normalization values")
        print("Normalizing recordings...")
        for user in self.data_dict['original'].keys():
            for rec in self.data_dict['original'][user].keys():
                for sensor in self.sensors:
                    if norm_range[sensor]:
                        cols = [col for col in self.data_columns if col.startswith(sensor)]
                        for col in cols:
                            self.data_dict['original'][user][rec][col] = \
                                (self.data_dict['original'][user][rec][col].values - sensor_shift[sensor]) / \
                                sensor_scale[sensor]
        self.sensor_shift = sensor_shift
        self.sensor_scale = sensor_scale
        print("Finished normalizing recordings")

    def normalize_global_2(self, norm_range=None):
        """
        The same as normalize_global but uses pre-calculated normalization values for speed purposes. The
        pre-calculated values can be stored under constants.py
        :param norm: See normalize_global
        :return: See normalize_global
        """
        if norm_range is None:
            norm = {sensor: True for sensor in self.sensors}
        sensor_shift = {'ACC': -90.83861868871521, 'GYRO': -32.85540978272818, 'LIGHT': np.nan,
                        'MAG': -184.9962120725973, 'PRESS': np.nan}
        sensor_scale = {'ACC': 185.55766760223042, 'GYRO': 58.16958899110071, 'LIGHT': np.nan,
                        'MAG': 577.7807153840964, 'PRESS': np.nan}
        print("Normalizing recordings...")
        for user in self.data_dict['original'].keys():
            for rec in self.data_dict['original'][user].keys():
                for sensor in self.sensors:
                    if norm_range[sensor]:
                        cols = [col for col in self.data_columns if col.startswith(sensor)]
                        for col in cols:
                            self.data_dict['original'][user][rec][col] = \
                                (self.data_dict['original'][user][rec][col].values - sensor_shift[sensor]) / \
                                sensor_scale[sensor]
        print("Finished normalizing recordings")


    def normalize_global_tanh(self, norm_range=None):
        """
        Normalize data globally using tanh scaling. Computes global min and max per sensor type
        and applies tanh normalization to scale values into [-1, 1].

        :param data_dict: Dictionary containing sensor data organized by user and recording.
        :param data_columns: List of sensor column names.
        :param sensors: List of sensor types (e.g., ['ACC', 'GYRO']).
        :param norm_range: Dictionary specifying which sensors to normalize.
        :return: Modified data_dict with normalized sensor data.
        """
        if norm_range is None:
            norm_range = {sensor: True for sensor in self.sensors}

        sensor_max = {key: -np.inf for key in self.sensors}
        sensor_min = {key: np.inf for key in self.sensors}
        
        print("Computing global min/max for tanh normalization...")
        # Step 1: Compute global min and max values
        for user in self.data_dict['original'].keys():
            for rec in self.data_dict['original'][user].keys():
                for sensor in self.sensors:
                    if norm_range.get(sensor, False):
                        cols = [col for col in self.data_columns if col.startswith(sensor)]
                        sensor_data = self.data_dict['original'][user][rec][cols].values
                        sensor_max[sensor] = np.max([sensor_max[sensor], np.max(sensor_data)])
                        sensor_min[sensor] = np.min([sensor_min[sensor], np.min(sensor_data)])
        sensor_shift = {key: sensor_min[key] for key in self.sensors}
        sensor_scale = {key: sensor_max[key] - sensor_min[key] for key in self.sensors}
        print("sensor_shift: ", sensor_shift)
        print("sensor_scale: ", sensor_scale)
        print("Applying tanh scaling...")
        sensor_center = {sensor: (sensor_max[sensor] + sensor_min[sensor]) / 2.0 for sensor in self.sensors}
        sensor_scale = {sensor: (sensor_max[sensor] - sensor_min[sensor]) / 2.0 for sensor in self.sensors}
        print("sensor_center:", sensor_center)
        print("sensor_scale: ", sensor_scale)

        print("Applying tanh normalization...")
        # Step 2: Apply tanh normalization
        for user in self.data_dict['original'].keys():
            for rec in self.data_dict['original'][user].keys():
                for sensor in self.sensors:
                    if norm_range.get(sensor, False):
                        cols = [col for col in self.data_columns if col.startswith(sensor)]
                        for col in cols:
                            centered = self.data_dict['original'][user][rec][col].values - sensor_center[sensor]
                            scaled = centered / (sensor_scale[sensor] + 1e-8)
                            self.data_dict['original'][user][rec][col] = np.tanh(scaled)

        self.sensor_shift = sensor_shift
        self.sensor_scale = sensor_scale
        self.sensor_center = sensor_center
        print("Finished applying tanh normalization.")

    def normalize_global_tanh_per_label(self, norm_range=None, per_label_axis=False):
        """
        Normalize data globally using tanh scaling. Computes global min and max per sensor type
        and applies tanh normalization to scale values into [-1, 1].
        
        :param norm_range: Dictionary specifying which sensors to normalize.
        :param per_label_axis: If True, uses axis=None for label=0 and axis=0 for labels 1-5.
                            If False, uses standard global normalization.
        :return: Modified data_dict with normalized sensor data.
        """
        if norm_range is None:
            norm_range = {sensor: True for sensor in self.sensors}

        if not per_label_axis:
            # Original implementation - global normalization
            sensor_max = {key: -np.inf for key in self.sensors}
            sensor_min = {key: np.inf for key in self.sensors}
            
            print("Computing global min/max for tanh normalization...")
            # Step 1: Compute global min and max values
            for user in self.data_dict['original'].keys():
                for rec in self.data_dict['original'][user].keys():
                    for sensor in self.sensors:
                        if norm_range.get(sensor, False):
                            cols = [col for col in self.data_columns if col.startswith(sensor)]
                            sensor_data = self.data_dict['original'][user][rec][cols].values
                            sensor_max[sensor] = np.max([sensor_max[sensor], np.max(sensor_data)])
                            sensor_min[sensor] = np.min([sensor_min[sensor], np.min(sensor_data)])
            
            sensor_center = {sensor: (sensor_max[sensor] + sensor_min[sensor]) / 2.0 for sensor in self.sensors}
            sensor_scale = {sensor: (sensor_max[sensor] - sensor_min[sensor]) / 2.0 for sensor in self.sensors}
            print("sensor_center:", sensor_center)
            print("sensor_scale: ", sensor_scale)

            print("Applying tanh normalization...")
            # Step 2: Apply tanh normalization
            for user in self.data_dict['original'].keys():
                for rec in self.data_dict['original'][user].keys():
                    for sensor in self.sensors:
                        if norm_range.get(sensor, False):
                            cols = [col for col in self.data_columns if col.startswith(sensor)]
                            for col in cols:
                                centered = self.data_dict['original'][user][rec][col].values - sensor_center[sensor]
                                scaled = centered / (sensor_scale[sensor] + 1e-8)
                                self.data_dict['original'][user][rec][col] = np.tanh(scaled)
        
        else:
            # Per-label normalization
            print("Computing per-label tanh normalization...")
            
            # First pass: compute statistics per label and sensor
            label_stats = {}
            for label in self.labels:
                label_stats[label] = {
                    sensor: {'max': -np.inf, 'min': np.inf, 'data_points': []} 
                    for sensor in self.sensors
                }
            
            # Collect data points for each label
            print("Collecting data points per label...")
            for user in self.data_dict['original'].keys():
                for rec in self.data_dict['original'][user].keys():
                    labels = self.data_dict['original'][user][rec]['label'].values
                    
                    for sensor in self.sensors:
                        if norm_range.get(sensor, False):
                            cols = [col for col in self.data_columns if col.startswith(sensor)]
                            sensor_data = self.data_dict['original'][user][rec][cols].values
                            
                            # Group by label
                            for label in self.labels:
                                label_mask = labels == label
                                if np.any(label_mask):
                                    label_data = sensor_data[label_mask]
                                    label_stats[label][sensor]['data_points'].append(label_data)
            
            # Compute statistics for each label
            print("Computing normalization statistics per label...")
            for label in self.labels:
                for sensor in self.sensors:
                    if norm_range.get(sensor, False):
                        if label_stats[label][sensor]['data_points']:
                            all_data = np.concatenate(label_stats[label][sensor]['data_points'], axis=0)
                            
                            if label == 0:
                                # For label 0: use axis=None (flatten all dimensions)
                                flat_data = all_data.flatten()
                                label_stats[label][sensor]['max'] = np.max(flat_data)
                                label_stats[label][sensor]['min'] = np.min(flat_data)
                            else:
                                # For labels 1-5: use axis=0 (compute per feature)
                                label_stats[label][sensor]['max'] = np.max(all_data, axis=0)
                                label_stats[label][sensor]['min'] = np.min(all_data, axis=0)
                            
                            # Clear data points to save memory
                            label_stats[label][sensor]['data_points'] = None
            
            # Second pass: apply normalization
            print("Applying per-label tanh normalization...")
            for user in self.data_dict['original'].keys():
                for rec in self.data_dict['original'][user].keys():
                    labels = self.data_dict['original'][user][rec]['label'].values
                    
                    for sensor in self.sensors:
                        if norm_range.get(sensor, False):
                            cols = [col for col in self.data_columns if col.startswith(sensor)]
                            
                            # Process each label segment
                            for label in self.labels:
                                label_mask = labels == label
                                if np.any(label_mask):
                                    indices = np.where(label_mask)[0]
                                    
                                    if label == 0:
                                        # For label 0: normalize with global stats (axis=None)
                                        center = (label_stats[label][sensor]['max'] + 
                                                label_stats[label][sensor]['min']) / 2.0
                                        scale = (label_stats[label][sensor]['max'] - 
                                                label_stats[label][sensor]['min']) / 2.0
                                        
                                        for col in cols:
                                            col_data = self.data_dict['original'][user][rec][col].values[indices]
                                            centered = col_data - center
                                            scaled = centered / (scale + 1e-8)
                                            self.data_dict['original'][user][rec].loc[self.data_dict['original'][user][rec].index[indices], col] = np.tanh(scaled)
                                    
                                    else:
                                        # For labels 1-5: normalize per feature (axis=0)
                                        max_vals = label_stats[label][sensor]['max']
                                        min_vals = label_stats[label][sensor]['min']
                                        
                                        for i, col in enumerate(cols):
                                            center = (max_vals[i] + min_vals[i]) / 2.0
                                            scale = (max_vals[i] - min_vals[i]) / 2.0
                                            
                                            col_data = self.data_dict['original'][user][rec][col].values[indices]
                                            centered = col_data - center
                                            scaled = centered / (scale + 1e-8)
                                            self.data_dict['original'][user][rec].loc[self.data_dict['original'][user][rec].index[indices], col] = np.tanh(scaled)
        
        print("Finished applying tanh normalization.")


    def load_data(self, data_path, data_columns, labels, stroke_labels=None, users=None):
        """
        Load processed swimming data.
        :param data_path: Path to processed swimming data.
        :param data_columns: Sensor columns that are used in the data set.
        :param labels: Labels that are read into the data windows
        :param stroke_labels: Stroke labels that are read into the data windows
        :param users: Users whose data is loaded. users=None means everybody is loaded.
        :return: A dictionary containing all data
        """
        # Ensure data_columns includes all sensor columns
        self.data_columns = data_columns
        self.data_columns_ix = {key: ix for (ix, key) in enumerate(data_columns)}
        self.sensors = list(np.unique([col.split('_')[0] for col in self.data_columns]))
        self.labels = list(np.sort(labels))
        self.labels_index = {label: ix for (ix, label) in enumerate(self.labels)}
        self.stroke_labels = stroke_labels

        # Create ordered columns list to match CSV header order
        if self.stroke_labels is not None:
            self.columns = ['row_index'] + ['timestamp'] + self.data_columns + ['label'] + self.stroke_labels
        else: 
            self.columns = ['row_index'] + ['timestamp'] + self.data_columns + ['label']

        #if self.stroke_labels is not None:
        #    self.columns.append('stroke_labels')

        self.mirror_ix = [ix for ix in range(len(data_columns)) if data_columns[ix] in constants.AXIS_MIRROR]
        self.users = users if users else utils.dirs_in_path(data_path)

        self.data_dict = {'original': {user: dict() for user in self.users}}
        for user in self.users:
            print(f"Loading user: {user}")
            user_path = os.path.join(data_path, user)
            csv_files = os.listdir(user_path)

            self.data_dict['original'][user] = {rec: None for rec in csv_files}
            for rec in csv_files:
                file_path = os.path.join(user_path, rec)
                df = pd.read_csv(file_path)

                # Check if 'row_index' already exists
                if 'row_index' not in df.columns:
                    df.insert(0, 'row_index', np.arange(len(df), dtype=np.int32))
                if self.stroke_labels is not None: 
                    # Ensure stroke labels are integers
                    if not pd.api.types.is_integer_dtype(df[self.stroke_labels]):
                        #print(f"Casting {self.stroke_labels} to integer for file {file_path}")
                        df[self.stroke_labels] = df[self.stroke_labels].astype(int)

                    # Check for missing stroke_labels
                    missing_stroke_labels = [col for col in self.stroke_labels if col not in df.columns]
                    if missing_stroke_labels:
                        raise ValueError(f"Missing stroke label columns {missing_stroke_labels} in {file_path}")

                # Reorder columns to match self.columns
                try:
                    df = df[self.columns]
                except KeyError as e:
                    print(f"Error in reordering columns for {file_path}: {e}")
                    continue

                self.data_dict['original'][user][rec] = df


    def augment_recordings(self, time_scale_factors=None):
        """
        Create augmented versions of recordings through time scaling
        :param time_scale_factors: A list of factors used to create time-scaled versions of original recordings
        :return: The new augmented versions are stored inside the data dictionary
        """
        if time_scale_factors is not None:
            self.time_scale_factors = time_scale_factors
            for factor in time_scale_factors:
                print("Augmenting with time-scale factor: %s" % factor)
                new_group = 'time_scaled_' + str(factor)
                self.data_dict[new_group] = {user: dict() for user in self.users}
                for user in self.users:
                    for rec in list(self.data_dict['original'][user].keys()):
                        self.data_dict[new_group][user][rec] = \
                            utils.time_scale_dataframe(self.data_dict['original'][user][rec], factor, 'timestamp', 'label', static_cols=['row_index'], binary_cols=['stroke_labels'])
                        # Debugging: Verify alignment
                        """
                        print(f"User: {user}, Recording: {rec}, Time-Scale Factor: {factor}")
                        print("Original Timestamps (first 5):", ", ".join(f"{x:.6f}" for x in self.data_dict['original'][user][rec]['timestamp'].head()))
                        print("Augmented Timestamps (first 5):", ", ".join(f"{x:.6f}" for x in self.data_dict[new_group][user][rec]['timestamp'].head()))
                        print(f"Original Stroke Indices: {np.where(self.data_dict['original'][user][rec]['stroke_labels'].values == 1)[0]}")
                        print(f"Augmented Stroke Indices: {np.where(self.data_dict[new_group][user][rec]['stroke_labels'].values == 1)[0]}")
                        """

        groups = list(self.data_dict.keys())
        self.group_probs = {group: 1/len(groups) for group in groups}

    def augment_stroke_labels(self, stroke_range=5):
        """
        Augments stroke labels in the dataset to include a range around detected peaks.

        Parameters:
            data_dict (dict): Dictionary containing the data, with keys for 'sensor_data' and 'stroke_labels'.
            label_range (int): Number of samples before and after each peak to also label as a stroke.

        Returns:
            dict: Updated data_dict with augmented stroke labels.
        """
        if stroke_range is not None:
            self.stroke_range = stroke_range
            groups = self.data_dict.keys()
            print("Augmenting with stroke_range: %s" % stroke_range)

            for group in groups:
                for user in self.users:

                    if user not in self.data_dict[group]:
                        print(f"Warning: User {user} not found in group {group}")
                        continue


                    for rec in self.data_dict[group][user].keys():
                        # Retrieve the stroke labels for the current record
                        stroke_labels = np.array(self.data_dict[group][user][rec]['stroke_labels'])
                        num_samples = len(stroke_labels)
                        
                        # Identify peaks (where stroke_labels == 1)
                        peak_indices = np.where(stroke_labels == 1)[0]
                        
                        # Augment stroke labels within the specified range around each peak
                        for peak_idx in peak_indices:
                            start_idx = max(0, peak_idx - stroke_range)
                            end_idx = min(num_samples, peak_idx + stroke_range + 1)
                            stroke_labels[start_idx:end_idx] = 1
                        
                        # Update the stroke labels in the data_dict
                        self.data_dict[group][user][rec]['stroke_labels'] = stroke_labels.tolist()
            
            print("Stroke label augmentation complete.")

    def augment_stroke_labels_to_heatmaps(self, stroke_sigma=3.0):
        """
        Replace the binary stroke_labels in self.data_dict with a Gaussian
        bump‐heatmap at each true event.  Must be called *before* compile_windows.

        Parameters
        ----------
        stroke_sigma : float
            Standard deviation of the Gaussian bump, in frames.
        """
        if stroke_sigma is not None:
            self.stroke_range = stroke_sigma
            groups = self.data_dict.keys()
            print(f"Augmenting stroke labels to heat‐maps with σ = {stroke_sigma} frames")

            # build a 1D Gaussian kernel (truncate at ±3σ)
            radius = int(np.ceil(3 * stroke_sigma))
            x = np.arange(-radius, radius+1, dtype=np.float32)
            kernel = np.exp(-0.5 * (x / stroke_sigma)**2)
        # kernel = kernel / kernel.sum()

            for group in groups:
                for user in self.users:
                    for rec in self.data_dict[group][user].keys():
                        # Retrieve the stroke labels for the current record
                        labels_bin = np.array(self.data_dict[group][user][rec]['stroke_labels'])

                        # convolve with gaussian, keep same length
                        heat = np.convolve(labels_bin, kernel, mode='same')

                        # clip to [0,1] in case overlapping bumps >1
                        heat = np.clip(heat, 0.0, 1.0)

                        # store back as floats (shape still (num_samples,))
                        self.data_dict[group][user][rec]['stroke_labels'] = heat.tolist()

        print("Stroke‐heatmap augmentation complete.")
        
    def sliding_window_locs(self, win_len, slide_len):
        """
        Compute sliding window start and stop sample index for each original and augmented recording
        :param win_len: Window length in number of samples
        :param slide_len: Slide length in number of samples
        :return: A dictionary with the same structure as the data dictionary but containing window start and stop
                 timestamps
        """
        self.win_len = win_len
        self.slide_len = slide_len
        groups = self.data_dict.keys()
        self.window_locs = {group: {user: dict() for user in self.users} for group in self.data_dict.keys()}
        for group in groups:
            for user in self.users:
                for rec in self.data_dict[group][user].keys():
                    x = np.arange(len(self.data_dict[group][user][rec]['timestamp'].values))
                    self.window_locs[group][user][rec] = \
                        segmentation.sliding_windows_start_stop(x=x, win_len=self.win_len,
                                                                slide_len=self.slide_len)

    def normalize_window(self, win_data, norm_type='statistical', axis=None):
        """
        Normalize the data of a window.
        :param win_data: A 2-D array of data. Columns are different sensors.
        :param norm_type: The normalization type employed: 'statistical', 'mean', 'statistical_combined', or 'tanh_scaled'.
        :return: The same win_data normalized appropriately.
        """
        if norm_type == 'statistical':
            win_data = (win_data - np.mean(win_data, axis=axis)) / np.std(win_data, axis=axis)
        elif norm_type == 'mean':
            win_data = win_data - np.mean(win_data, axis=axis)
        elif norm_type == 'statistical_combined':
            win_data = win_data - np.mean(win_data, axis=axis)
            for sensor in self.sensors:
                sensor_cols_ix = np.where(self.data_columns.startswith(sensor))
                std_sensor = np.mean(np.std(win_data[:, sensor_cols_ix], axis=axis))
                win_data[:, sensor_cols_ix] = win_data[:, sensor_cols_ix] / std_sensor
        elif norm_type == 'tanh_scaled':
            # Scale data to [-1, 1] range
            min_val = np.min(win_data, axis=axis)
            max_val = np.max(win_data, axis=axis)
            win_data = 2 * (win_data - min_val) / (max_val - min_val) - 1
        elif norm_type == 'zscore_tanh':
            mean_val = np.mean(win_data, axis=axis)
            std_val = np.std(win_data, axis=axis) + 1e-8  # Avoid division by zero
            win_data = (win_data - mean_val) / std_val
            win_data = np.tanh(win_data)

            # Final correction: Subtract mean after tanh
            corrected_mean = np.mean(win_data, axis=axis)
            win_data -= corrected_mean  # Explicitly zero-center each channel
        elif norm_type == None:
            return win_data
        else:
            raise ValueError("Invalid window normalization type")
        return win_data


    def get_window_label(self, win_labels, label_type='proportional', multi_class_thresh=0, majority_thresh=0):
        """
        Get an array of categorized labels
        :param win_labels: An array of sample-by-sample labels
        :param label_type: The type of labeling employed: 'proportional', 'majority', 'multi_class' or 'sparse'
        :param multi_class_thresh: Only used if label_type='multi_class'. The proportion of samples that are needed
                                   such that a label is considered as part of the window
        :param majority_thresh: Only used if label_type='majority'. The proportion of samples needed such that the
                                window is labeled of that type.
        :return: An array of categorized labels
        """
        win_labels_list, label_count = np.unique(win_labels, return_counts=True)
        if len(set(win_labels_list) - set(self.labels)) > 0:
            # Ignore windows that contain labels that are not included
            return None, None
        majority_label = win_labels_list[np.argmax(label_count)]
        win_label_cat = np.zeros(len(self.labels), dtype=np.float32)
        for (ii, label) in enumerate(win_labels_list):
           # win_label_cat[self.labels_index[label]] = label_count[ii] / self.win_len
            win_label_cat[self.labels_index[label]] = label_count[ii] / np.sum(label_count)  # Normalize

        if label_type == 'proportional':
            return win_label_cat, majority_label
        elif label_type == 'multi_class':
            #ix_above_thresh = np.where(win_label_cat - multi_class_thresh >= 0)[0]
            ix_above_thresh = np.where(win_label_cat >= multi_class_thresh)[0]

            if len(ix_above_thresh) == 0:
                # Ignore window with no clear label
                return None, None
            else:
                """
                win_label_cat = np.zeros(len(self.labels), dtype=int)
                win_label_cat[ix_above_thresh] = 1
                win_label_cat = win_label_cat / np.sum(win_label_cat)
                return win_label_cat, majority_label
                """
                win_label_cat = np.zeros(len(self.labels), dtype=np.float32)
                win_label_cat[ix_above_thresh] = 1
                win_label_cat /= np.sum(win_label_cat)  # Ensure sum is 1
                return win_label_cat, majority_label
        elif label_type == 'majority':
            
            if np.max(win_label_cat) > majority_thresh:
                ix_majority = np.argmax(win_label_cat)
                win_label_cat = np.zeros(len(self.labels), dtype=np.float32)
                win_label_cat[ix_majority] = 1
                return win_label_cat, majority_label
            
            #if np.max(win_label_cat) > majority_thresh:
             #   ix_majority = np.argmax(win_label_cat)
              #  win_label_cat = np.zeros(len(self.labels), dtype=np.float32)
              #  win_label_cat[ix_majority] = 1  # Hard one-hot only if dominant
              #  return win_label_cat, majority_label
            else:
                # Ignore window if no clear majority
                return None, None
        elif label_type == 'sparse':
            # Return a single integer label for the majority class
            return win_label_cat, majority_label
        elif label_type == 'soft_one_hot':
            total = np.sum(win_label_cat)
            if total > 0:
                win_label_cat /= total  # Normalize safely
                return win_label_cat, majority_label
            else:
                return None, None  # If no valid label, discard
        elif label_type == 'raw':
            return win_label_cat, majority_label
        else:
            raise ValueError("Invalid labeling type")

    def compile_windows(self, norm_type='statistical', label_type='proportional', multi_class_thresh=0.2,
                        majority_thresh=0):
        """
        Compile windows based on window and slide lengths.
        :param norm_type: Window normalization type: 'statistical', 'statistical_combined' , 'tanh_scaled' or 'mean'
        :param label_type: Labeling type: 'majority', 'proportional', 'multi_class', or 'sparse'
        :param multi_class_thresh: Threshold above which a label is assigned under multi class labeling. Only used if
                                    label_type = 'multi_class'
        :param majority_thresh: A threshold for the minimum proportion of the majority label in a window
        :return: self.data_windows created
        """
        groups = self.data_dict.keys()
        # Adjust the data structure to include both sparse and one-hot labels
        self.data_windows = {group: {label: dict() for label in self.labels} for group in groups}
        for group in groups:
            for user in self.users:
                print(f"Compiling windows: {group}, {user}")
                temp_user_windows = {
                    label: {
                        'data': np.zeros((10000, self.win_len, len(self.data_columns)), dtype=np.float32),
                        'raw_label': np.zeros((10000, self.win_len), dtype=np.int32), # For raw labels used in GAN
                        'sparse_label': np.zeros(10000, dtype=np.int32),  # 1D array for sparse labels
                        'one_hot_label': np.zeros((10000, len(self.labels)), dtype=np.float32),  # For one-hot labels
                        'stroke_labels': np.zeros((10000, self.win_len, len(self.stroke_labels)), dtype=np.int32)  # Ensure integer dtype
                    }
                    for label in self.labels
                }
                cnt_user_label = {label: 0 for label in self.labels}
                if user not in self.data_dict[group]:
                    print(f"Warning: User {user} not found in group {group}")
                    continue
                for rec in self.data_dict[group][user].keys():
                    for i in range(len(self.window_locs[group][user][rec][0])):
                        win_start = self.window_locs[group][user][rec][0][i]
                        win_stop = self.window_locs[group][user][rec][1][i]
                        # Extract sensor data
                        win_data = self.data_dict[group][user][rec][self.data_columns].values[win_start: win_stop+1]

                        # Extract labels
                        win_labels = self.data_dict[group][user][rec]['label'].values[win_start: win_stop+1]
                        win_label_cat, majority_label = self.get_window_label(
                            win_labels, label_type=label_type,
                            multi_class_thresh=multi_class_thresh,
                            majority_thresh=majority_thresh
                        )

                        # Extract stroke labels
                        win_stroke_labels = self.data_dict[group][user][rec][self.stroke_labels].values[win_start: win_stop+1]

                        # Check if there is any stroke in the window (optional)
                        if np.any(win_stroke_labels == 1):
                            # Adjust label if majority is 0 but a stroke exists with a valid minority label
                            if majority_label == 0 and np.any(win_labels != 0):
                                # Assign the minority label associated with the stroke
                                #minority_label = np.argmax(np.bincount(win_labels[win_labels != 0].astype(int)))
                                #majority_label = minority_label
                                win_stroke_labels = np.zeros_like(win_stroke_labels)
                                
                        if majority_label == 0:
                            # Normalize sensor data other than style = 0(rest)
                            win_data = self.normalize_window(win_data, norm_type=norm_type, axis=None)
                        else:
                            # Normalize sensor data
                            win_data = self.normalize_window(win_data, norm_type=norm_type, axis=0)

                        if win_label_cat is None or majority_label is None:
                            # A "bad" window was returned
                            continue

                        # Assign data and labels to temporary structures
                        temp_user_windows[majority_label]['data'][cnt_user_label[majority_label]] = win_data
                        temp_user_windows[majority_label]['raw_label'][cnt_user_label[majority_label]] = win_labels
                        temp_user_windows[majority_label]['sparse_label'][cnt_user_label[majority_label]] = majority_label
                        temp_user_windows[majority_label]['one_hot_label'][cnt_user_label[majority_label]] = win_label_cat
                        temp_user_windows[majority_label]['stroke_labels'][cnt_user_label[majority_label]] = win_stroke_labels

                        cnt_user_label[majority_label] += 1

                # Strip away from temp_user_windows and save in data_windows
                for label in self.labels:
                    if cnt_user_label[label] == 0:
                        continue
                    else:

                        self.data_windows[group][label][user] = {
                            'data': np.copy(temp_user_windows[label]['data'][0:cnt_user_label[label]]),
                            'raw_label': np.copy(temp_user_windows[label]['raw_label'][0:cnt_user_label[label]]),
                            'sparse_label': np.copy(temp_user_windows[label]['sparse_label'][0:cnt_user_label[label]]),
                            'one_hot_label': np.copy(temp_user_windows[label]['one_hot_label'][0:cnt_user_label[label]]),
                            'stroke_labels': np.copy(temp_user_windows[label]['stroke_labels'][0:cnt_user_label[label]])
                        }

        self.norm_type = norm_type
        self.label_type = label_type
        self.multi_class_thresh = multi_class_thresh
        self.majority_thresh = majority_thresh


    def compute_folded_weights_per_user_old(self, train_dict):
        """
        Compute folded weights for each user in the training set based on the number of windows available.
        :param train_dict: Dictionary with keys corresponding to labels and values of users
        :return: Dictionary of folded weights for each user
        """
        # Initialize weights for each user
        folded_weights_for_user = {}

        for label, users in train_dict.items():
            for user in users:
                # Count number of windows for user in original group and label
                n_samples = len(self.data_windows['original'][label][user]['data'])
                if n_samples > 0:
                    folded_weights_for_user[user] = 1.0 / n_samples
                else:
                    folded_weights_for_user[user] = 0.0  # or skip user

        # Normalize weights so sum to 1
        total_weight = sum(folded_weights_for_user.values())
        if total_weight > 0:
            for user in folded_weights_for_user:
                folded_weights_for_user[user] /= total_weight
        print(f"Sum of folded weights before normalization: {total_weight}")
        print(f"Sum of folded weights after normalization: {sum(folded_weights_for_user.values())}")

        return folded_weights_for_user

    def compute_folded_weights_per_user(self, train_dict):
        """
        Compute folded weights for each user in the training set based on the number of windows available,
        normalized so that the mean sample weight is 1.
        :param train_dict: Dictionary with keys corresponding to labels and values of users
        :return: Dictionary of folded weights for each user
        """
        folded_weights_for_user = {}
        user_sample_counts = {}

        # First, accumulate total sample counts per user across all labels
        for label, users in train_dict.items():
            for user in users:
                n_samples = len(self.data_windows['original'][label][user]['data'])
                user_sample_counts[user] = user_sample_counts.get(user, 0) + n_samples

        # Compute unnormalized folded weights (inverse of sample counts)
        for user, count in user_sample_counts.items():
            folded_weights_for_user[user] = 1.0 / count if count > 0 else 0.0

        # Compute total number of samples
        total_samples = sum(user_sample_counts.values())

        # Compute weighted sum of folded weights times sample counts
        weighted_sum = sum(folded_weights_for_user[user] * user_sample_counts[user] for user in user_sample_counts)

        # Normalize folded weights so that mean sample weight = 1
        if weighted_sum > 0:
            for user in folded_weights_for_user:
                folded_weights_for_user[user] *= total_samples / weighted_sum

        print(f"Sum of folded weights times sample counts (before normalization): {weighted_sum}")
        print(f"Sum of folded weights (after normalization): {sum(folded_weights_for_user.values())}")
        print(f"Mean sample weight after normalization: {sum(folded_weights_for_user[user] * user_sample_counts[user] for user in user_sample_counts) / total_samples}")

        return folded_weights_for_user

    def compute_folded_weights(self, train_dict):
        """
        Compute per-sample folded weights for the training data.
        :param train_dict: Dict {label: [users]} defining which users to include.
        :return: np.array of sample weights aligned with concatenated training windows.
        """
        x_train = None
        folds = {'start': [], 'stop': [], 'size': []}
        labels = list(train_dict.keys())
        cnt = 0

        # First, concatenate all user windows sequentially and record slices
        for label in labels:
            for user in train_dict[label]:
                if user in self.data_windows['original'][label]:
                    x_new = self.data_windows['original'][label][user]['data']
                    if x_train is None:
                        x_train = x_new
                        folds['start'].append(0)
                        folds['stop'].append(len(x_new))
                    else:
                        x_train = np.concatenate((x_train, x_new), axis=0)
                        folds['start'].append(folds['stop'][cnt - 1])
                        folds['stop'].append(folds['start'][cnt] + len(x_new))
                    folds['size'].append(len(x_new))
                    cnt += 1

        # Create sample weights array (same length as x_train)
        sample_weights = np.zeros(shape=(len(x_train),), dtype=np.float32)

        # Assign each sample weight = inverse of number of samples from that user's chunk
        for i in range(len(folds['start'])):
            start = folds['start'][i]
            stop = folds['stop'][i]
            size = folds['size'][i]
            if size > 0:
                sample_weights[start:stop] = 1.0 / size
            else:
                sample_weights[start:stop] = 0.0

        # Normalize to average weight of 1 (optional but recommended)
        sample_weights = sample_weights / np.sum(sample_weights) * len(sample_weights)

        return sample_weights


    def batch_generator_dicts(
        self, 
        train_dict, 
        batch_size=64, 
        noise_std=None, 
        mirror_prob=None, 
        random_rot_deg=30, 
        use_4D=False, 
        swim_style_output=True, 
        stroke_label_output=True,
        return_stroke_mask=False,
        compute_sample_weights=False,  
        global_swim_weights=None,  
        global_stroke_weights=None,
        folded_weights_for_user=None,
        return_combined_labels=False
    ):
        """
        A generator that yields a random set of windows with optional stroke mask and sample weights
        
        :param train_dict: Dictionary with keys corresponding to labels and values of users
        :param batch_size: Number of windows yielded
        :param noise_std: Optional noise standard deviation 
        :param mirror_prob: Optional probability of mirroring axes
        :param return_stroke_mask: Whether to return stroke mask as a separate output
        :param compute_sample_weights: Whether to compute sample weights for the outputs
        :param global_swim_weights: Pre-computed global weights for swim styles
        :param global_stroke_weights: Pre-computed global weights for stroke labels
        :return: batch_data and corresponding labels/weights based on specified parameters
        """
        groups = list(self.data_dict.keys())
        group_probs = [self.group_probs[group] for group in groups]
        while True:
            batch_data = np.zeros((batch_size, self.win_len, len(self.data_columns)), dtype=np.float32)
            batch_raw_labels = np.zeros((batch_size, self.win_len), dtype=np.int32)  
            batch_labels = np.zeros(batch_size, dtype=np.int32)
            batch_labels_cat = np.zeros((batch_size, len(self.labels)), dtype=np.float32)
            batch_stroke_labels = np.zeros((batch_size, self.win_len, len(self.stroke_labels)), dtype=np.int32)
            stroke_mask = np.ones((batch_size, self.win_len, len(self.stroke_labels)), dtype=np.int32)  # Shape (batch_size, 180, 1)
            batch_swim_weights = np.zeros(batch_size, dtype=np.float32)
            batch_stroke_weights = np.zeros((batch_size, self.win_len, 1), dtype=np.float32)

            for i in range(batch_size):
                r_group = np.random.choice(groups, p=group_probs)
                r_label = np.random.choice(self.labels)

                # Filter users who have data for the current label
                available_users = [user for user in train_dict[r_label] if user in self.data_windows[r_group][r_label]]
                if not available_users:
                    print(f"No available users for group {r_group} and label {r_label}")
                    continue  # Skip this iteration if no users have data for this label
                
                if compute_sample_weights:
                    if folded_weights_for_user is not None:
                        user_weights = np.array([folded_weights_for_user.get(user, 1.0) for user in available_users], dtype=np.float32)
                        user_weights_sum = user_weights.sum()
                        if user_weights_sum == 0:
                            print(f"Warning: sum of folded weights for available users is zero at batch index {i}")
                            user_weights = np.ones_like(user_weights) / len(user_weights)
                        else:
                            user_weights /= user_weights_sum
                        r_user = np.random.choice(available_users, p=user_weights)
                        folded_w = folded_weights_for_user.get(r_user, 1.0)
                        batch_swim_weights[i] = folded_w
                        batch_stroke_weights[i, :, 0] = folded_w
                        # Debug print per sample
                       # print(f"Batch idx {i}: sampled user {r_user} with folded weight {folded_w:.4f}")

                    else:
                        r_user = np.random.choice(available_users)
                else:
                    r_user = np.random.choice(available_users)
                
                #r_user = np.random.choice(available_users)

                r_win = np.random.choice(len(self.data_windows[r_group][r_label][r_user]['data']))
                batch_data[i, :, :] = self.data_windows[r_group][r_label][r_user]['data'][r_win]
                batch_raw_labels[i] = self.data_windows[r_group][r_label][r_user]['raw_label'][r_win]
                batch_labels[i] = self.data_windows[r_group][r_label][r_user]['sparse_label'][r_win]
                batch_labels_cat[i] = self.data_windows[r_group][r_label][r_user]['one_hot_label'][r_win]
                batch_stroke_labels[i] = self.data_windows[r_group][r_label][r_user]['stroke_labels'][r_win]


                # Apply mask: Ignore stroke labels for transition windows
                if r_label == 0:  # Transition label
                    stroke_mask[i, :, :] = 0  # Mask out stroke labels for this window


            # Apply Mirroring
            if mirror_prob is not None:
                mirror_samples = np.random.choice([True, False], p=[mirror_prob, 1 - mirror_prob], size=batch_size)
                if any(mirror_samples) is True:
                    batch_data[mirror_samples][:, self.mirror_ix] = -batch_data[mirror_samples][:,  self.mirror_ix]
            # Apply Rotation
            if random_rot_deg is not None:
                r_theta = np.random.uniform(-random_rot_deg/180*np.pi, random_rot_deg/180*np.pi, batch_size)
                for i in range(batch_size):
                    for sensor in self.sensors:
                        if sensor in ['ACC', 'GYRO']:
                            ix_1 = self.data_columns_ix[sensor + '_1']
                            ix_2 = self.data_columns_ix[sensor + '_2']
                            b1 = np.copy(batch_data[i, :, ix_1])
                            b2 = np.copy(batch_data[i, :, ix_2])
                            batch_data[i, :, ix_1] = b1 * np.cos(r_theta[i]) - b2 * np.sin(r_theta[i])
                            batch_data[i, :, ix_2] = b1 * np.sin(r_theta[i]) + b2 * np.cos(r_theta[i])
            # Apply Noise
            if noise_std is not None:
                batch_data = batch_data + np.random.normal(0, noise_std, batch_data.shape)

            # Combine sensor data and stroke labels along the feature axis
            #batch_data_combined = np.concatenate((batch_data, batch_stroke_labels), axis=2)

            # Create a copy of batch_data_combined for the stroke_label_output
          #  batch_data_combined_stroke = np.copy(batch_data_combined)

            # Reshape to 4D for CNN Training
            if use_4D:
                #batch_data_combined = batch_data_combined.reshape((batch_data_combined.shape[0], batch_data_combined.shape[1], batch_data_combined.shape[2], 1))
               # batch_data_combined_stroke = batch_data_combined_stroke.reshape((batch_data_combined_stroke.shape[0], batch_data_combined_stroke.shape[1], batch_data_combined_stroke.shape[2], 1))
                batch_data = batch_data.reshape((batch_data.shape[0], batch_data.shape[1], batch_data.shape[2], 1))
            # Yield options based on label type and output requirements
            # Compute sample weights if requested
            # After batch is filled, debug summary of folded weights assigned
         #   if compute_sample_weights and folded_weights_for_user is not None:
          #      print(f"Batch folded swim weights: min={batch_swim_weights.min():.4f}, max={batch_swim_weights.max():.4f}, mean={batch_swim_weights.mean():.4f}")
          #      print(f"Batch folded stroke weights: min={batch_stroke_weights.min():.4f}, max={batch_stroke_weights.max():.4f}, mean={batch_stroke_weights.mean():.4f}")

            sample_weights = {}
            if compute_sample_weights:
                # Swim style sample weights
                if swim_style_output:
                    if global_swim_weights is not None:
                        # Use global weights
                        #swim_weight_map = dict(enumerate(global_swim_weights))
                        #global_swim_weights_array = np.array([global_swim_weights[i] for i in range(len(global_swim_weights))])

                        if self.label_type == 'sparse':
                            # Map each label to its global weight
                            swim_style_weights = np.array([global_swim_weights[int(label)] for label in batch_labels])
                        elif self.label_type == 'raw':
                            swim_style_weights = np.array([global_swim_weights[int(label)] for label in batch_raw_labels])
                        else:  # categorical
                            # Get class indices from one-hot encoded labels
                            label_indices = np.argmax(batch_labels_cat, axis=1)
                            #swim_style_weights = np.array([global_swim_weights[int(label)] for label in label_indices])
                            swim_style_weights = global_swim_weights[label_indices]  # shape (batch_size,)

                    else:
                        # Use per-batch computation (existing code)
                        if self.label_type == 'sparse': 
                            swim_style_weights = compute_sample_weight('balanced', y=batch_labels)
                        elif self.label_type == 'raw':
                            swim_style_weights = compute_sample_weight('balanced', y=batch_raw_labels)  
                        else:  # categorical
                            label_indices = np.argmax(batch_labels_cat, axis=1)
                            swim_style_weights = compute_sample_weight('balanced', y=label_indices)
                    
                 #   if folded_weights_for_user is not None:
                      #  swim_style_weights_before = swim_style_weights.copy()

                  #      swim_style_weights *= batch_swim_weights
                        #final_swim_weights = final_swim_weights / np.mean(final_swim_weights)
                      #  print(f"Swim style weights before folded weighting: min={swim_style_weights_before.min():.4f}, max={swim_style_weights_before.max():.4f}, mean={swim_style_weights_before.mean():.4f}")
                       # print(f"Swim style weights after folded weighting: min={swim_style_weights.min():.4f}, max={swim_style_weights.max():.4f}, mean={swim_style_weights.mean():.4f}")
                    
                    sample_weights['swim_style_output'] = swim_style_weights.astype(np.float32)
                
                # Stroke detection sample weights
                if stroke_label_output:
                    if global_stroke_weights is not None:
                        # Use global weights - map each class to its global weight
                        #stroke_weight_map = dict(enumerate(global_stroke_weights))
                        
                        # Create empty array for weights
                        stroke_weights = np.zeros_like(batch_stroke_labels, dtype=np.float32)
                        # Remove last dim to get shape (batch_size, 180)
                        stroke_class_indices = batch_stroke_labels.squeeze(-1).astype(int)  # shape (64, 180)
                      #  global_stroke_weights_array = np.array([global_stroke_weights[0], global_stroke_weights[1]], dtype=np.float32)

                        # Map each label (0 or 1) to its global weight
                        stroke_weights = global_stroke_weights[stroke_class_indices]  # shape (64, 180)

                        # Add back the last dimension if needed to match model output shape
                        stroke_weights = stroke_weights[..., np.newaxis]  # shape (64, 180, 1)

                        # Map each stroke label to its global weight
                     #   for i in range(batch_stroke_labels.shape[0]):
                      #      for j in range(batch_stroke_labels.shape[1]):
                       #         for k in range(batch_stroke_labels.shape[2]):
                        #            label = int(batch_stroke_labels[i, j, k])
                         #           stroke_weights[i, j, k] = global_stroke_weights[label]

                    else:
                        # Use per-batch computation (existing code)
                        flat_stroke_labels = batch_stroke_labels.reshape(-1)
                        flat_weights = compute_sample_weight('balanced', y=flat_stroke_labels)
                        stroke_weights = flat_weights.reshape(batch_stroke_labels.shape)
                    
                   # if folded_weights_for_user is not None:
                       # stroke_weights_before = stroke_weights.copy()

                      #  stroke_weights *= batch_stroke_weights
                     #   print(f"Stroke weights before folded weighting: min={stroke_weights_before.min():.4f}, max={stroke_weights_before.max():.4f}, mean={stroke_weights_before.mean():.4f}")
                      #  print(f"Stroke weights after folded weighting: min={stroke_weights.min():.4f}, max={stroke_weights.max():.4f}, mean={stroke_weights.mean():.4f}")
                    
                    sample_weights['stroke_label_output'] = stroke_weights.astype(np.float32)

            # Yield options based on label type and output requirements
            if self.label_type == 'sparse':
                if swim_style_output and stroke_label_output:
                    if compute_sample_weights:
                        yield (
                            batch_data,
                            {
                                'swim_style_output': batch_labels, 
                                'stroke_label_output': batch_stroke_labels
                            },
                            sample_weights
                        )
                    elif return_stroke_mask:
                        yield (
                            batch_data,
                            {
                                'swim_style_output': batch_labels, 
                                'stroke_label_output': batch_stroke_labels
                            },
                            {
                                'stroke_label_output': stroke_mask
                            }
                        )
                    else:                         
                        yield (
                            batch_data,
                            {
                                'swim_style_output': batch_labels, 
                                'stroke_label_output': batch_stroke_labels
                            }
                        )
                elif swim_style_output:
                    if compute_sample_weights:
                        yield (
                            batch_data, 
                            batch_labels,
                            sample_weights.get('swim_style_output')
                        )
                    else:
                        yield (
                            batch_data, batch_labels
                        )
                else:
                    if compute_sample_weights:
                        yield (
                            batch_data, 
                            batch_stroke_labels,
                            sample_weights.get('stroke_label_output')
                        )
                    elif return_stroke_mask:
                        yield (
                            batch_data, batch_stroke_labels, stroke_mask
                        )
                    else:
                        yield (
                            batch_data,batch_stroke_labels
                        )
            elif self.label_type == 'majority':  # Categorical labels
                if swim_style_output and stroke_label_output:
                    if compute_sample_weights:
                        yield (
                            batch_data,
                            {
                                'swim_style_output': batch_labels_cat, 
                                'stroke_label_output': batch_stroke_labels
                            },
                            { 'swim_style_output': sample_weights.get('swim_style_output'), 'stroke_label_output': sample_weights.get('stroke_label_output')}
                        )
                    elif return_stroke_mask:
                        yield (
                            batch_data,
                            {
                                'swim_style_output': batch_labels_cat, 
                                'stroke_label_output': batch_stroke_labels
                            },
                            {
                                'stroke_label_output': stroke_mask
                            }
                        )
                    else:
                        yield (
                            batch_data,
                            {
                                'swim_style_output': batch_labels_cat, 
                                'stroke_label_output': batch_stroke_labels
                            }
                        )
                elif swim_style_output:
                    if compute_sample_weights:
                        yield (
                            batch_data, 
                            batch_labels_cat,
                            sample_weights.get('swim_style_output')
                        )
                    else:
                        yield (
                            batch_data, batch_labels_cat
                        )
                else:
                    if compute_sample_weights:
                        yield (
                            batch_data, 
                            batch_stroke_labels,
                            sample_weights.get('stroke_label_output')
                        )
                    elif return_stroke_mask:
                        yield (
                            batch_data, batch_stroke_labels, stroke_mask
                        )
                    else:
                        yield (
                            batch_data, batch_stroke_labels
                        )
            else:  # Raw labels
                if swim_style_output and stroke_label_output:
                    if compute_sample_weights:
                        yield (
                            batch_data,
                            {
                                'swim_style_output': batch_labels_cat, 
                                'stroke_label_output': batch_stroke_labels
                            },
                            sample_weights
                        )
                    elif return_stroke_mask:
                        yield (
                            batch_data,
                            {
                                'swim_style_output': batch_raw_labels, 
                                'stroke_label_output': batch_stroke_labels
                            },
                            {
                                'stroke_label_output': stroke_mask
                            }
                        )
                    else:
                        if return_combined_labels:
                            batch_raw_labels_3d = np.expand_dims(batch_raw_labels, axis=-1) #S
                            batch_labels_combined = np.concatenate((batch_raw_labels_3d, batch_stroke_labels), axis=-1, dtype=np.float32)
                            yield (
                                batch_data, batch_labels_combined
                            )
                        else: 
                            yield (
                                batch_data,
                                {
                                    'swim_style_output': batch_raw_labels, 
                                    'stroke_label_output': batch_stroke_labels
                                }
                            )
                elif swim_style_output:
                    if compute_sample_weights:
                        yield (
                            batch_data, 
                            batch_labels_cat,
                            sample_weights.get('swim_style_output')
                        )
                    yield (
                        batch_data, batch_raw_labels
                    )
                else:
                    if compute_sample_weights:
                        yield (
                            batch_data, 
                            batch_stroke_labels,
                            sample_weights.get('stroke_label_output')
                        )
                    elif return_stroke_mask:
                        yield (
                            batch_data, batch_stroke_labels, stroke_mask
                        )
                    else:
                        yield (
                            batch_data, batch_stroke_labels
                        )

    def get_windows(self, users, group='original'):
        """
        Get all windows from a set of users.
        :param users: A list of users
        :return: x_val: A 3-dimensional numpy array of window data
                y_val_cat: A 2-dimensional numpy array of corresponding one-hot encoded labels
                y_stroke_val: A 3-dimensional numpy array of corresponding stroke labels
        """
        x_val = None
        y_val_raw_new = None
        y_val_sparse = None
        y_val_cat = None
        y_stroke_val = None

        for label in self.labels:
            for user in users:
                if user in self.data_windows[group][label].keys():
                    # Extract data, swim style labels, and stroke labels
                    x_val_new = self.data_windows[group][label][user]['data']
                    y_val_raw_new = self.data_windows[group][label][user]['raw_label']
                    y_val_sparse_new = self.data_windows[group][label][user]['sparse_label']
                    y_val_cat_new = self.data_windows[group][label][user]['one_hot_label']
                    y_stroke_val_new = self.data_windows[group][label][user]['stroke_labels']

                    # Initialize or concatenate
                    if x_val is None:
                        x_val = x_val_new
                        y_val_raw = y_val_raw_new
                        y_val_sparse = y_val_sparse_new
                        y_val_cat = y_val_cat_new
                        y_stroke_val = y_stroke_val_new
                    else:
                        x_val = np.concatenate((x_val, x_val_new), axis=0)
                        y_val_raw = np.concatenate((y_val_raw, y_val_raw_new), axis=0)
                        y_val_sparse = np.concatenate((y_val_sparse, y_val_sparse_new), axis=0)
                        y_val_cat = np.concatenate((y_val_cat, y_val_cat_new), axis=0)
                        y_stroke_val = np.concatenate((y_stroke_val, y_stroke_val_new), axis=0)

        return x_val, y_val_raw, y_val_sparse, y_val_cat, y_stroke_val


    def get_windows_dict(self, label_user_dict, return_weights=False, return_mask=False, transition_label=0, return_raw_labels=False):
        """
        Get all windows from a set of users.

        Args:
            label_user_dict (dict): A dictionary of labels (keys) and users (values).
            return_weights (bool): Whether to return the sample weights.
            return_mask (bool): Whether to return a mask for stroke labels.
            transition_label (int): Label used for transitions, which will be masked.

        Returns:
            x_val: A 3-dimensional numpy array of window data
            y_val_sparse: A 2-dimensional numpy array of corresponding majority labels.
            y_val_cat: A 2-dimensional numpy array of corresponding one-hot encoded labels.
            y_stroke_val: A 3-dimensional numpy array of corresponding stroke labels.
            swim_style_sample_weights (optional): Weight for each sample (if return_weights is True).
            stroke_mask (optional): Mask for stroke labels to mask counting 0 strokes in transition(combined 0,5) class (if return_mask is True).
        """
        x_val = None
        y_val_raw_new = None
        y_val_sparse = None
        y_val_cat = None
        y_stroke_val = None
        folds = {'start': [], 'stop': [], 'size': []}
        labels = list(label_user_dict.keys())
        cnt = 0

        for label in labels:
            for user in label_user_dict[label]:
                if user in self.data_windows['original'][label].keys():
                    # Extract data, swim style labels, and stroke labels
                    x_val_new = self.data_windows['original'][label][user]['data']
                    y_val_raw_new = self.data_windows['original'][label][user]['raw_label']
                    y_val_sparse_new = self.data_windows['original'][label][user]['sparse_label']
                    y_val_cat_new = self.data_windows['original'][label][user]['one_hot_label']
                    y_stroke_val_new = self.data_windows['original'][label][user]['stroke_labels']

                    # Initialize or concatenate
                    if x_val is None:
                        x_val = x_val_new
                        y_val_raw = y_val_raw_new
                        y_val_sparse = y_val_sparse_new
                        y_val_cat = y_val_cat_new
                        y_stroke_val = y_stroke_val_new
                        folds['start'].append(0)
                        folds['stop'].append(len(x_val_new))
                    else:
                        x_val = np.concatenate((x_val, x_val_new), axis=0)
                        y_val_raw = np.concatenate((y_val_raw, y_val_raw_new), axis=0)
                        y_val_sparse = np.concatenate((y_val_sparse, y_val_sparse_new), axis=0)
                        y_val_cat = np.concatenate((y_val_cat, y_val_cat_new), axis=0)
                        y_stroke_val = np.concatenate((y_stroke_val, y_stroke_val_new), axis=0)
                        folds['start'].append(folds['stop'][cnt - 1])
                        folds['stop'].append(folds['start'][cnt] + len(x_val_new))

                    folds['size'].append(len(x_val_new))
                    cnt += 1

        # Combine sensor data and stroke labels
        #x_val_combined = np.concatenate((x_val, y_stroke_val), axis=2)
        # Create a masked version of x_val_combined
        #y_val_combined = np.copy(x_val_combined)

        #results = [x_val_combined, y_val_cat, y_stroke_val, y_val_combined]
        results = [x_val, y_val_sparse, y_val_cat, y_stroke_val]#, y_val_combined]

        if return_weights:
            swim_style_sample_weights = np.zeros(len(x_val))
            for i in range(len(folds['start'])):
                swim_style_sample_weights[folds['start'][i]:folds['stop'][i]] = 1 / folds['size'][i]
            swim_style_sample_weights = swim_style_sample_weights / np.sum(swim_style_sample_weights) * len(swim_style_sample_weights)
            results.append(swim_style_sample_weights)

        if return_mask:
            # Create stroke mask and exclude transition labels
            #stroke_mask = np.ones_like(y_val_combined)  # Initialize stroke mask as a copy of combined data
            stroke_mask = np.ones((len(x_val), y_stroke_val.shape[1], 1), dtype=np.int32)  # (batch_size, 180, 1)

            if transition_label in label_user_dict.keys():
                transition_indices = np.where(y_val_cat[:, self.labels_index[transition_label]] > 0)[0]
                for idx in transition_indices:
                    stroke_mask[idx, :, :] = 0  # Mask the combined data for transition labels          

            results.append(stroke_mask)

        if return_raw_labels:
            results.append(y_val_raw)

        return tuple(results)


    def get_windows_dict_stratified(self, label_user_dict, return_weights=False, return_mask=False, transition_label=0):
        """
        Get all windows from a set of users with stratified sampling.

        Args:
            label_user_dict (dict): A dictionary of labels (keys) and users (values).
            return_weights (bool): Whether to return the sample weights.
            return_mask (bool): Whether to return a mask for stroke labels.
            transition_label (int): Label used for transitions, which will be masked.

        Returns:
            x_val: A 3-dimensional numpy array of window data
            y_val_sparse: A 2-dimensional numpy array of corresponding majority labels.
            y_val_cat: A 2-dimensional numpy array of corresponding one-hot encoded labels.
            y_stroke_val: A 3-dimensional numpy array of corresponding stroke labels.
            swim_style_sample_weights (optional): Weight for each sample (if return_weights is True).
            stroke_mask (optional): Mask for stroke labels to mask counting 0 strokes in transition(combined 0,5) class (if return_mask is True).
        """
        x_val = []
        y_val_sparse = []
        y_val_cat = []
        y_stroke_val = []
        folds = {'start': [], 'stop': [], 'size': []}
        labels = list(label_user_dict.keys())
        cnt = 0

        # Extract swim style labels for stratified sampling
        swim_style_labels = []
        for label in labels:
            for user in label_user_dict[label]:
                if user in self.data_windows['original'][label].keys():
                    swim_style_labels.append(self.data_windows['original'][label][user]['swim_style'])

        # Stratified K-Folds
        skf = StratifiedKFold(n_splits=5, shuffle=True)

        # Iterate through the stratified folds
        for train_index, val_index in skf.split(np.zeros(len(swim_style_labels)), swim_style_labels):
            for i in val_index:
                label = labels[i]
                user = label_user_dict[label][i]

                if user in self.data_windows['original'][label].keys():
                    # Extract data, swim style labels, and stroke labels
                    x_val_new = self.data_windows['original'][label][user]['data']
                    y_val_sparse_new = self.data_windows['original'][label][user]['sparse_label']
                    y_val_cat_new = self.data_windows['original'][label][user]['one_hot_label']
                    y_stroke_val_new = self.data_windows['original'][label][user]['stroke_labels']

                    # Initialize or concatenate
                    if not x_val:
                        x_val = x_val_new
                        y_val_sparse = y_val_sparse_new
                        y_val_cat = y_val_cat_new
                        y_stroke_val = y_stroke_val_new
                        folds['start'].append(0)
                        folds['stop'].append(len(x_val_new))
                    else:
                        x_val = np.concatenate((x_val, x_val_new), axis=0)
                        y_val_sparse = np.concatenate((y_val_sparse, y_val_sparse_new), axis=0)
                        y_val_cat = np.concatenate((y_val_cat, y_val_cat_new), axis=0)
                        y_stroke_val = np.concatenate((y_stroke_val, y_stroke_val_new), axis=0)
                        folds['start'].append(folds['stop'][cnt - 1])
                        folds['stop'].append(folds['start'][cnt] + len(x_val_new))

                    folds['size'].append(len(x_val_new))
                    cnt += 1

        # Combine sensor data and stroke labels
        results = [x_val, y_val_sparse, y_val_cat, y_stroke_val]

        if return_weights:
            swim_style_sample_weights = np.zeros(len(x_val))
            for i in range(len(folds['start'])):
                swim_style_sample_weights[folds['start'][i]:folds['stop'][i]] = 1 / folds['size'][i]
            swim_style_sample_weights = swim_style_sample_weights / np.sum(swim_style_sample_weights) * len(swim_style_sample_weights)
            results.append(swim_style_sample_weights)

        if return_mask:
            stroke_mask = np.ones((len(x_val), y_stroke_val.shape[1], 1), dtype=np.int32)  # (batch_size, 180, 1)

            if transition_label in label_user_dict.keys():
                transition_indices = np.where(y_val_cat[:, self.labels_index[transition_label]] > 0)[0]
                for idx in transition_indices:
                    stroke_mask[idx, :, :] = 0  # Mask the combined data for transition labels          

            results.append(stroke_mask)

        return tuple(results)

    def create_custom_user(self, label_user_dict, name='custom'):
        """
        Create a custom user based on label-user combinations
        :param label_user_dict: A dictionary of label (key) and user (value) combinations
        :param name: The name given to the custom user
        :return: The custom user is created within the object. The originals are removed.
        """
        for style in label_user_dict.keys():
            for user in label_user_dict[style]:
                for group in self.data_windows.keys():
                    if name not in self.data_windows[group][style].keys():
                        self.data_windows[group][style][name] = self.data_windows[group][style][user]
                    else:
                        x_new = self.data_windows[group][style][user]['data']
                        y_sparse_new = self.data_windows[group][style][user]['sparse_label']
                        y_cat_new = self.data_windows[group][style][user]['one_hot_label']
                        stroke_labels_new = self.data_windows[group][style][user]['stroke_labels']

                        self.data_windows[group][style][name]['data'] = \
                            np.concatenate((self.data_windows[group][style][name]['data'], x_new))
                        self.data_windows[group][style][name]['sparse_label'] = \
                            np.concatenate((self.data_windows[group][style][name]['sparse_label'], y_sparse_new))
                        self.data_windows[group][style][name]['one_hot_label'] = \
                            np.concatenate((self.data_windows[group][style][name]['one_hot_label'], y_cat_new))
                        self.data_windows[group][style][name]['stroke_labels'] = \
                            np.concatenate((self.data_windows[group][style][name]['stroke_labels'], stroke_labels_new))

                    self.data_windows[group][style].pop(user, None)

        # Add the custom user to the users list if not already present
        if name not in self.users:
            self.users.append(name)

    def combine_labels(self, labels, new_label):
        """
        Combine labels under a new label
        :param labels: A list of labels
        :param new_label: A value for the new label
        :return: self.data_dict['original'] and self.labels have been appropriately modified
        """
        for user in self.data_dict['original'].keys():
            for rec in self.data_dict['original'][user].keys():
                current_labels = self.data_dict['original'][user][rec]['label'].values
                ix = np.where(current_labels == labels[0])[0]
                for label in labels[1:]:
                    ix = np.concatenate([ix, np.where(current_labels == label)[0]])
                if len(ix) == 0:
                    continue
                self.data_dict['original'][user][rec]['label'].values[ix] = new_label
        for label in labels:
            self.labels.remove(label)
        self.labels.append(new_label)
        self.labels = list(np.sort(self.labels))
        self.labels_index = {label: ix for (ix, label) in enumerate(self.labels)}

    def draw_train_val_dicts(self, users_train, users_per_class=None, manual_val_dict=None):
        """
        Draw a set of train/validation label-user combinations.
        :param users_train: List of users in the full training set.
        :param users_per_class: Number of users for each label in the validation set (optional).
        :param manual_val_dict: Manually-defined validation dictionary (optional).
        :return: train_dict: A dictionary of label-user combinations used in the training set.
                val_dict: A dictionary of label-user combinations used in the validation set.
        """
        val_dict = {l: None for l in self.labels}
        train_dict = {l: None for l in self.labels}

        for l in self.labels:
            users_in_class = list(self.data_windows['original'][l].keys())
            users_train_in_class = [u for u in users_in_class if u in users_train]

            # Use manually-defined val_dict if provided
            if manual_val_dict and l in manual_val_dict:
                val_dict[l] = manual_val_dict[l]
            else:
                # Otherwise, randomly select validation users
                if type(users_per_class) is dict:
                    val_dict[l] = np.random.choice(users_train_in_class, users_per_class[l], replace=False).tolist()
                else:
                    val_dict[l] = np.random.choice(users_train_in_class, users_per_class, replace=False).tolist()

            # Exclude validation users from the training set
            train_dict[l] = [u for u in users_train_in_class if u not in val_dict[l]]

        return train_dict, val_dict

    def stratified_loso_split(self):
        all_users = set()
        for label in self.labels:
            all_users.update(self.data_windows['original'][label].keys())
        all_users = sorted(all_users)

        splits = []
        for val_user in all_users:
            train_users = [u for u in all_users if u != val_user]

            train_dict = {label: [] for label in self.labels}
            val_dict = {label: [] for label in self.labels}

            for label in self.labels:
                # Assign val_user to val_dict if present
                if val_user in self.data_windows['original'][label]:
                    val_dict[label].append(val_user)

                # Assign train users to train_dict if present
                for u in train_users:
                    if u in self.data_windows['original'][label]:
                        train_dict[label].append(u)

            splits.append((train_dict, val_dict))
        return splits


    def preprocess_filtering(self, butter_params=None, savgol_params=None, apply_butter=True, apply_savgol=True):
        """
        Apply filtering (Butterworth and/or Savitzky-Golay) to the raw sensor data.

        Parameters:
        -----------
        butter_params : dict, optional
            Dictionary with parameters for the Butterworth filter:
                - 'cutoff': Cutoff frequency (default: 0.3)
                - 'fs': Sampling frequency (default: 30)
                - 'order': Filter order (default: 4)
        savgol_params : dict, optional
            Dictionary with parameters for the Savitzky-Golay filter:
                - 'window_size': Window size (default: 5)
                - 'poly_order': Polynomial order (default: 2)
        apply_butter : bool
            Whether to apply the Butterworth filter (default: True).
        apply_savgol : bool
            Whether to apply the Savitzky-Golay filter (default: True).

        Returns:
        --------
        None
        """
        if not apply_butter and not apply_savgol:
            print("No filters selected. Exiting preprocessing...")
            return
        # Default parameters
        butter_params = butter_params or {'cutoff': 0.3, 'fs': 30, 'order': 4}
        savgol_params = savgol_params or {'window_size': 5, 'poly_order': 2}

        def butter_highpass_filter(data, cutoff, fs, order):
            nyquist = 0.5 * fs
            normal_cutoff = cutoff / nyquist
            b, a = scipy.signal.butter(order, normal_cutoff, btype='high', analog=False)
            return scipy.signal.filtfilt(b, a, data)

        def smooth_data_with_savgol(data, window_size, poly_order):
            window_size = max(3, window_size if window_size % 2 == 1 else window_size + 1)
            return scipy.signal.savgol_filter(data, window_length=window_size, polyorder=poly_order)

        print(
            f"Applying {'Butterworth ' if apply_butter else ''}"
            f"{'and ' if apply_butter and apply_savgol else ''}"
            f"{'Savitzky-Golay ' if apply_savgol else ''}filtering to sensor data..."
        )
        for user in self.users:
            for rec in self.data_dict['original'][user].keys():
                df = self.data_dict['original'][user][rec]
                for sensor in self.sensors:
                    sensor_cols = [col for col in self.data_columns if col.startswith(sensor)]
                    for col in sensor_cols:
                        filtered_data = df[col].values
                        
                        # Apply Butterworth filter if enabled
                        if apply_butter:
                            filtered_data = butter_highpass_filter(
                                filtered_data,
                                butter_params['cutoff'],
                                butter_params['fs'],
                                butter_params['order']
                            )
                        
                        # Apply Savitzky-Golay filter if enabled
                        if apply_savgol:
                            filtered_data = smooth_data_with_savgol(
                                filtered_data,
                                savgol_params['window_size'],
                                savgol_params['poly_order']
                            )
                        
                        # Update the DataFrame with filtered data
                        df[col] = filtered_data

        print("Finished filtering sensor data.")

    def augment_windows(self, label_type='proportional', multi_class_thresh=0.2, majority_thresh=0, noise_std=None, mirror_prob=None, random_rot_deg=30):
        """
        Augment windows based on window and slide lengths.
        :param norm_type: Window normalization type: 'statistical', 'statistical_combined' , 'tanh_scaled' or 'mean'
        :param label_type: Labeling type: 'majority', 'proportional', 'multi_class', 'sparse' or 'raw'
        :return: self.data_windows created
        """
        groups = self.data_dict.keys()
        # Adjust the data structure to include both sparse and one-hot labels
        self.data_windows = {group: {label: dict() for label in self.labels} for group in groups}
        for group in groups:
            for user in self.users:
                group_mirror = False
                if mirror_prob is not None:
                    group_mirror = np.random.rand() < mirror_prob


                # Build a list of active augmentations
                augmentations = []
                if mirror_prob:
                    augmentations.append("Mirroring")
                if random_rot_deg:
                    augmentations.append("Rotation")
                if noise_std:
                    augmentations.append("Noise")
              #  if time_scale and time_scale != 1.0:
               #     augmentations.append(f"Time Scale ({time_scale:.2f}x)")

                # Join into readable string
                aug_str = ", ".join(augmentations) if augmentations else "No augmentations"

                print(f"Augmenting windows: {group}, {user} with {aug_str} applied to sensor data...")


                temp_user_windows = {
                    label: {
                        'data': np.zeros((10000, self.win_len, len(self.data_columns)), dtype=np.float32),
                        'raw_label': np.zeros((10000, self.win_len), dtype=np.int32), # For raw labels used in GAN
                        'sparse_label': np.zeros(10000, dtype=np.int32),  # 1D array for sparse labels
                        'one_hot_label': np.zeros((10000, len(self.labels)), dtype=np.float32),  # For one-hot labels
                        'stroke_labels': np.zeros((10000, self.win_len, len(self.stroke_labels)), dtype=np.int32)  # Ensure integer dtype
                    }
                    for label in self.labels
                }
                cnt_user_label = {label: 0 for label in self.labels}
                if user not in self.data_dict[group]:
                    print(f"Warning: User {user} not found in group {group}")
                    continue
                for rec in self.data_dict[group][user].keys():
                    for i in range(len(self.window_locs[group][user][rec][0])):
                        win_start = self.window_locs[group][user][rec][0][i]
                        win_stop = self.window_locs[group][user][rec][1][i]
                        # Extract sensor data
                        win_data = self.data_dict[group][user][rec][self.data_columns].values[win_start: win_stop+1]

                        # Extract labels
                        win_labels = self.data_dict[group][user][rec]['label'].values[win_start: win_stop+1]
                        win_label_cat, majority_label = self.get_window_label(
                            win_labels, label_type=label_type,
                            multi_class_thresh=multi_class_thresh,
                            majority_thresh=majority_thresh
                        )

                        # Extract stroke labels
                        win_stroke_labels = self.data_dict[group][user][rec][self.stroke_labels].values[win_start: win_stop+1]

                        # Check if there is any stroke in the window (optional)
                        if np.any(win_stroke_labels == 1):
                            # Adjust label if majority is 0 but a stroke exists with a valid minority label
                            if majority_label == 0 and np.any(win_labels != 0):
                                # Assign the minority label associated with the stroke
                                #minority_label = np.argmax(np.bincount(win_labels[win_labels != 0].astype(int)))
                                #majority_label = minority_label
                                win_stroke_labels = np.zeros_like(win_stroke_labels)
                                

                        # Apply Mirroring
                        """                       
                        if mirror_prob is not None:
                            mirror_samples = np.random.choice([True, False], p=[mirror_prob, 1 - mirror_prob], size=self.win_len)
                            if any(mirror_samples) is True:
                                win_data[mirror_samples][:, self.mirror_ix] = -win_data[mirror_samples][:,  self.mirror_ix]
                        """
                        # Apply Mirroring: mirror all axes consistently for this group if selected
                        if group_mirror:
                            win_data[:, self.mirror_ix] *= -1

                        # Apply Rotation
                        if random_rot_deg is not None:
                            r_theta = np.random.uniform(-random_rot_deg/180*np.pi, random_rot_deg/180*np.pi, self.win_len)
                            for i in range(self.win_len):
                                for sensor in self.sensors:
                                    if sensor in ['ACC', 'GYRO']:
                                        ix_1 = self.data_columns_ix[sensor + '_1']
                                        ix_2 = self.data_columns_ix[sensor + '_2']
                                        b1 = np.copy(win_data[i, ix_1])
                                        b2 = np.copy(win_data[i, ix_2])
                                        win_data[i, ix_1] = b1 * np.cos(r_theta[i]) - b2 * np.sin(r_theta[i])
                                        win_data[i, ix_2] = b1 * np.sin(r_theta[i]) + b2 * np.cos(r_theta[i])
                        # Apply Noise
                        if noise_std is not None:
                            win_data = win_data + np.random.normal(0, noise_std, win_data.shape)

                        if win_label_cat is None or majority_label is None:
                            # A "bad" window was returned
                            continue

                        # Assign data and labels to temporary structures
                        temp_user_windows[majority_label]['data'][cnt_user_label[majority_label]] = win_data
                        temp_user_windows[majority_label]['raw_label'][cnt_user_label[majority_label]] = win_labels
                        temp_user_windows[majority_label]['sparse_label'][cnt_user_label[majority_label]] = majority_label
                        temp_user_windows[majority_label]['one_hot_label'][cnt_user_label[majority_label]] = win_label_cat
                        temp_user_windows[majority_label]['stroke_labels'][cnt_user_label[majority_label]] = win_stroke_labels

                        cnt_user_label[majority_label] += 1

                # Strip away from temp_user_windows and save in data_windows
                for label in self.labels:
                    if cnt_user_label[label] == 0:
                        continue
                    else:

                        self.data_windows[group][label][user] = {
                            'data': np.copy(temp_user_windows[label]['data'][0:cnt_user_label[label]]),
                            'raw_label': np.copy(temp_user_windows[label]['raw_label'][0:cnt_user_label[label]]),
                            'sparse_label': np.copy(temp_user_windows[label]['sparse_label'][0:cnt_user_label[label]]),
                            'one_hot_label': np.copy(temp_user_windows[label]['one_hot_label'][0:cnt_user_label[label]]),
                            'stroke_labels': np.copy(temp_user_windows[label]['stroke_labels'][0:cnt_user_label[label]])
                        }
        self.label_type = label_type
        self.multi_class_thresh = multi_class_thresh
        self.majority_thresh = majority_thresh

    def augment_recording_windows(self, label_type='proportional', multi_class_thresh=0.2, majority_thresh=0, noise_std=None, mirror_prob=None, random_rot_deg=30):
        groups = self.data_dict.keys()
        self.data_windows = {group: {label: dict() for label in self.labels} for group in groups}
        
        for group in groups:
            for user in self.users:
                if user not in self.data_dict[group]:
                    print(f"Warning: User {user} not found in group {group}")
                    continue

                for rec in self.data_dict[group][user].keys():
                    # Recording-level mirroring decision
                    rec_mirror = False
                    if mirror_prob is not None:
                        rec_mirror = np.random.rand() < mirror_prob

                    # Log augmentations
                    augmentations = []
                    if rec_mirror:
                        augmentations.append("Mirroring")
                    if random_rot_deg:
                        augmentations.append("Rotation")
                    if noise_std:
                        augmentations.append("Noise")
                    aug_str = ", ".join(augmentations) if augmentations else "No augmentations"
                    print(f"Augmenting: {group}/{user}/{rec} with {aug_str}")

                    temp_user_windows = {
                        label: {
                            'data': np.zeros((10000, self.win_len, len(self.data_columns)), dtype=np.float32),
                            'raw_label': np.zeros((10000, self.win_len), dtype=np.int32),
                            'sparse_label': np.zeros(10000, dtype=np.int32),
                            'one_hot_label': np.zeros((10000, len(self.labels)), dtype=np.float32),
                            'stroke_labels': np.zeros((10000, self.win_len, len(self.stroke_labels)), dtype=np.int32)
                        }
                        for label in self.labels
                    }
                    cnt_user_label = {label: 0 for label in self.labels}

                    for i in range(len(self.window_locs[group][user][rec][0])):
                        win_start = self.window_locs[group][user][rec][0][i]
                        win_stop = self.window_locs[group][user][rec][1][i]
                        win_data = self.data_dict[group][user][rec][self.data_columns].values[win_start: win_stop+1]
                        win_labels = self.data_dict[group][user][rec]['label'].values[win_start: win_stop+1]
                        win_stroke_labels = self.data_dict[group][user][rec][self.stroke_labels].values[win_start: win_stop+1]

                        win_label_cat, majority_label = self.get_window_label(
                            win_labels, label_type=label_type,
                            multi_class_thresh=multi_class_thresh,
                            majority_thresh=majority_thresh
                        )
                        if win_label_cat is None or majority_label is None:
                            continue

                        if np.any(win_stroke_labels == 1):
                            if majority_label == 0 and np.any(win_labels != 0):
                                win_stroke_labels = np.zeros_like(win_stroke_labels)

                        if rec_mirror:
                            win_data[:, self.mirror_ix] *= -1

                        if random_rot_deg is not None:
                            r_theta = np.random.uniform(-random_rot_deg / 180 * np.pi, random_rot_deg / 180 * np.pi, self.win_len)
                            for j in range(self.win_len):
                                for sensor in self.sensors:
                                    if sensor in ['ACC', 'GYRO']:
                                        ix_1 = self.data_columns_ix[sensor + '_1']
                                        ix_2 = self.data_columns_ix[sensor + '_2']
                                        b1 = np.copy(win_data[j, ix_1])
                                        b2 = np.copy(win_data[j, ix_2])
                                        win_data[j, ix_1] = b1 * np.cos(r_theta[j]) - b2 * np.sin(r_theta[j])
                                        win_data[j, ix_2] = b1 * np.sin(r_theta[j]) + b2 * np.cos(r_theta[j])

                        if noise_std is not None:
                            win_data += np.random.normal(0, noise_std, win_data.shape)

                        idx = cnt_user_label[majority_label]
                        temp_user_windows[majority_label]['data'][idx] = win_data
                        temp_user_windows[majority_label]['raw_label'][idx] = win_labels
                        temp_user_windows[majority_label]['sparse_label'][idx] = majority_label
                        temp_user_windows[majority_label]['one_hot_label'][idx] = win_label_cat
                        temp_user_windows[majority_label]['stroke_labels'][idx] = win_stroke_labels
                        cnt_user_label[majority_label] += 1

                    for label in self.labels:
                        if cnt_user_label[label] > 0:
                            self.data_windows[group][label].setdefault(user, {})
                            self.data_windows[group][label][user][rec] = {
                                'data': np.copy(temp_user_windows[label]['data'][:cnt_user_label[label]]),
                                'raw_label': np.copy(temp_user_windows[label]['raw_label'][:cnt_user_label[label]]),
                                'sparse_label': np.copy(temp_user_windows[label]['sparse_label'][:cnt_user_label[label]]),
                                'one_hot_label': np.copy(temp_user_windows[label]['one_hot_label'][:cnt_user_label[label]]),
                                'stroke_labels': np.copy(temp_user_windows[label]['stroke_labels'][:cnt_user_label[label]])
                            }

        self.label_type = label_type
        self.multi_class_thresh = multi_class_thresh
        self.majority_thresh = majority_thresh

    def get_recording_windows(self, users, group='original', rec=None):
        """
        Get all windows for specific users in a given group.
        If `rec` is provided, returns only data from that recording.

        :param users: List of user IDs
        :param group: Data group to use ('original', 'time_scaled_0.9', etc.)
        :param rec: If provided, restricts to this single recording per user
        :return: sensor_data, y_val_raw, y_val_sparse, y_val_cat, y_stroke_val
        """
        x_val = None
        y_val_raw = None
        y_val_sparse = None
        y_val_cat = None
        y_stroke_val = None

        for label in self.labels:
            for user in users:
                if user not in self.data_windows[group][label]:
                    continue

                recs_to_check = [rec] if rec else self.data_windows[group][label][user].keys()

                for recording in recs_to_check:
                    if recording not in self.data_windows[group][label][user]:
                        continue

                    data_entry = self.data_windows[group][label][user][recording]
                    x_val_new = data_entry['data']
                    y_val_raw_new = data_entry['raw_label']
                    y_val_sparse_new = data_entry['sparse_label']
                    y_val_cat_new = data_entry['one_hot_label']
                    y_stroke_val_new = data_entry['stroke_labels']

                    if x_val is None:
                        x_val = x_val_new
                        y_val_raw = y_val_raw_new
                        y_val_sparse = y_val_sparse_new
                        y_val_cat = y_val_cat_new
                        y_stroke_val = y_stroke_val_new
                    else:
                        x_val = np.concatenate((x_val, x_val_new), axis=0)
                        y_val_raw = np.concatenate((y_val_raw, y_val_raw_new), axis=0)
                        y_val_sparse = np.concatenate((y_val_sparse, y_val_sparse_new), axis=0)
                        y_val_cat = np.concatenate((y_val_cat, y_val_cat_new), axis=0)
                        y_stroke_val = np.concatenate((y_stroke_val, y_stroke_val_new), axis=0)

        return x_val, y_val_raw, y_val_sparse, y_val_cat, y_stroke_val

    def get_style_distribution(self, users, group='original', return_counts=False, label_names=constants.LABEL_NAMES):
        """
        Compute swim style distribution using flattened `raw_label` values (i.e., true per-sample labels).

        :param users: List of user IDs
        :param group: Data group to use ('original', 'time_scaled_0.9', etc.)
        :param return_counts: If True, return counts as dict instead of printing
        :param label_names: Optional label name map (e.g., {0: 'Null', 1: 'Freestyle', ...})
        :return: Dictionary of {label: count}
        """
        from collections import defaultdict

        label_counts = defaultdict(int)

        for label in self.labels:
            for user in users:
                user_data = self.data_windows[group][label].get(user)
                if user_data is None:
                    continue

                raw_label_windows = user_data.get('raw_label')  # shape: (num_windows, win_len)
                if isinstance(raw_label_windows, np.ndarray):
                    flattened = raw_label_windows.flatten()
                    unique, counts = np.unique(flattened, return_counts=True)
                    for u, c in zip(unique, counts):
                        label_counts[u] += c
                else:
                    print(f"⚠️ Skipping user={user}, label={label} — 'raw_label' missing or invalid")

        if return_counts:
            return dict(label_counts)

        # Print formatted output
        print(f"\nPer-sample Label Distribution (flattened raw_label) for group: '{group}'")
        total = sum(label_counts.values())
        for label in sorted(label_counts.keys()):
            label_str = label_names[label] if label_names else str(label)
            count = label_counts[label]
            pct = 100.0 * count / total if total else 0.0
            print(f"Label {label_str}: {count} samples ({pct:.2f}%)")

        return dict(label_counts)

def main():
    print("Running main")

if __name__ == '__main__':
    main()
