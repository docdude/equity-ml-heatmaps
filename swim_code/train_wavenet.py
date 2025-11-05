# A basic tutorial in how I load data and train a model
import wavenet_model
import utils
import utils_plot
from utils_train import CombinedEarlyStopping, CombinedMetricCallback, get_callbacks_combined, compile_model
from utils_peak_detect_v2 import EnhancedPeakConsistencyCallback
import learning_data
import os
import random as rn
import tensorflow as tf
import numpy as np
import pickle
import datetime
from tensorboard.plugins.hparams import api as hp
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
#tf.config.run_functions_eagerly(True)  # Run eagerly for CPU
#tf.data.experimental.enable_debug_mode()  # Help identify input pipeline issues
# A path to re-sampled recordings which are organized into folders by user name.
data_path = '/teamspace/studios/this_studio/SwimmingModelPython/swim_v2/data_modified_users_unfiltered'

# Path to where we want to save the training results
run_name = 'wavenet_14_unfiltered_zeronorm_axisnone_combine_heatmaps_85_1_cust_bfce_cce'
base_path = '/teamspace/studios/this_studio/SwimmingModelPython/swim_v2'
save_path = os.path.join(base_path, f'run_{run_name}')

# A list of user names which are loaded.

users_all = utils.folders_in_path(data_path)
users = [u for u in users_all] #if u not in users_ignore]
users.sort(key=int)

# Keeping it simple. Comment this out and use the code above if you want to load everybody
#users = ['2','6','7','11']
users_test = users

# List of users we want to train a model for
#users_test = ['30']

# Hyper-parameters for loading data.
data_parameters = {'users':                users,   # Users whose data is loaded
                   'labels':               [0, 1, 2, 3, 4, 5],  # Labels we want to use
                   'combine_labels':       {0: [0, 5]},     # Labels we want to combine. Here I am combining NULL and
                                                            # TURN into NULL
                   'data_columns':         ['ACC_0', 'ACC_1', 'ACC_2', 'GYRO_0', 'GYRO_1', 'GYRO_2'],
                   'stroke_labels': ['stroke_labels'],  # Binary stroke labels: 0 for no stroke, 1 for stroke
                   'time_scale_factors':   [0.9, 1.1],  # time-scaling factors we want to use. A copy is made of each
                                                        # recording with these factors.
                   'stroke_range':         None,       # Augments stroke labels in the dataset to include a range around detected peaks
                   'stroke_sigma':          2.0,
                   'win_len':              180,     # The length of the segmentation window in number of samples
                   'slide_len':            30,      # The slide length used for segmentation
                   'window_normalization': 'tanh_scaled',   # How we want to normalize the windows. Statistical means
                                                            # zero-mean and unit variance for each signal
                   'label_type':           'majority',  # How we label windows.
                   'majority_thresh':      0.75,    # Proportion of samples in a window that have to have the same label
                   'validation_set': {
                        0: 1,  # Transition (NULL + TURN combined as 0)
                        1: 1,  # Freestyle
                        2: 1,  # Breaststroke
                        3: 1,  # Backstroke
                        4: 1,  # Butterfly
                      #  5: 1,  # Turn
                   },
                   'debug':               False,
                   'debug_plot':           False
                   }

# Data is loaded and stored in this object
swimming_data = learning_data.LearningData()

# Load recordings from data_path. Recordings are stored under
# swimming_data.data_dict['original][user_name][recording_name] which is a Pandas DataFrame
swimming_data.load_data(data_path=data_path,
                        data_columns=data_parameters['data_columns'],
                        users=data_parameters['users'],
                        labels=data_parameters['labels'],
                        stroke_labels=data_parameters['stroke_labels'])
# Combine labels moved after compile_windows to allow style 0 normaliztion(axis=none)
if data_parameters['combine_labels'] is not None:
    for label in data_parameters['combine_labels'].keys():
        swimming_data.combine_labels(labels=data_parameters['combine_labels'][label], new_label=label)

# Data augmentation for recordings. This is only for time-scaling. Other data augmentations happen during the learning
# Stored under swimming_data['time_scaled_1.1'][user_name]...
swimming_data.augment_recordings(time_scale_factors=data_parameters['time_scale_factors'])

# Augments stroke labels in the dataset to include a range around detected peaks
#swimming_data.augment_stroke_labels(stroke_range=data_parameters['stroke_range'])
swimming_data.augment_stroke_labels_to_heatmaps(stroke_sigma=data_parameters['stroke_sigma'])
# Debug visualization time auementation of sensor data and stroke data
if data_parameters['debug_plot']:
    utils_plot.plot_style_specific_axes(swimming_data)


# Compute the locations of the sliding windows in each recording
swimming_data.sliding_window_locs(win_len=data_parameters['win_len'], slide_len=data_parameters['slide_len'])

# Compile the windows. Stored under swimming_data.data_windows[group][label][user]['data' or 'one_hot_label' or 'sparse_label' or 'stroke_label']
# Recordings are still stored under swimming_data.data_dict so a lot of memory might be needed
swimming_data.compile_windows(norm_type=data_parameters['window_normalization'],
                              label_type=data_parameters['label_type'],
                              majority_thresh=data_parameters['majority_thresh'])

# Combine labels
#if data_parameters['combine_labels'] is not None:
 #   for label in data_parameters['combine_labels'].keys():
  #      swimming_data.combine_labels(labels=data_parameters['combine_labels'][label], new_label=label)


# Parameters for training the CNN model
training_parameters = {'swim_style_lr':   4.14E-05,  # Constant for swim style
                       'stroke_lr':       1.07E-05,
                    #   'common_lr':       2.0e-5,
                       'beta_1':          0.9,
                       'beta_2':          0.999,
                       'batch_size':      128,
                       'max_epochs':      175,      # Keeping small for quick testing
                       'steps_per_epoch': 100,      # Keeping small for quick testing
                       'noise_std':       0.01,    # Noise standard deviation for data augmentation
                       'mirror_prob':     0.5,     # Probability of reversing a window for data augmentation
                       'random_rot_deg':  30,      # [-30, 30] is the range of rotation degrees we sample for each
                                                   # window in the mini-batch
                       'group_probs':     {'original': 0.7, 'time_scaled_0.9': 0.15, 'time_scaled_1.1': 0.15},
                       'sample_weights':   'sklearn',    # Sample weights for validation set, 'sklearn' or 'inverse_freq'
                       'stroke_mask':     False,    # Whether to use a mask for stroke labels
                       'labels':          swimming_data.labels,
                       'stroke_labels':   swimming_data.stroke_labels,
                       'stroke_label_output':       True,
                       'swim_style_output':         True,
                       'output_bias':               None,
                       'compute_sample_weights':    True
                       }

loss_parameters = {'alpha': 0.75,  
                   'gamma': 1.05,  # Focal loss parameters
                   'target_height': 0.85,  # Target height for the heatmap
                   'peak_weight': 12.0,  # Peak weight for the heatmap
                   'peak_threshold': 0.55,  # Threshold for peak detection in the heatmap
                   }

# The input shape of the Wavenet model
input_shape = (data_parameters['win_len'], len(data_parameters['data_columns']))


# Train all models
for (i, user_test) in enumerate(users_test):
    # Random seed stuff. Maybe this is overkill
    os.environ['PYTHONHASHSEED'] = '0'
    rn.seed(1337)
    np.random.seed(1337)
    tf.random.set_seed(1337)

    # Path for saving results
    print("Running experiment: %s" % user_test)
    experiment_save_path = os.path.join(save_path, user_test)

    # A path to log directory for Tensorboard
    log_dir = f"logs/wave_heat/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_run-{user_test}-{run_name}"
    if os.path.exists(experiment_save_path):
        print(f"Skipping: {user_test}, already processed.")
        continue
    else:
        os.makedirs(experiment_save_path)  # Creates all parent directories if they don't exist

    # Users whose data we use for training
    users_train = [u for u in users if u != user_test]

    # Draw users for each class. train_dict and val_dict are dictionaries whose keys are labels and they contain
    # lists of names for each label
    train_dict, val_dict = swimming_data.draw_train_val_dicts(users_train,
                                                              users_per_class=data_parameters['validation_set'],
                                                              manual_val_dict=None)
  # if data_parameters['debug']:
    print("Training dictionary: %s" % train_dict)   
    print("Validation dictionary: %s" % val_dict)


    # Calculate stroke label distribution for training set (excluding label 0)
    training_probabilities, training_mean, training_bias, training_stroke_sample_weights, training_stroke_class_weights = utils.calculate_stroke_label_distribution(
        label_user_dict=train_dict,
        swimming_data=swimming_data,
        data_type="training",
        exclude_label=None
    )
    training_parameters['output_bias'] = training_bias
    # Calculate stroke label distribution for validation set (excluding label 0)
    validation_probabilities, validation_mean, validation_bias, validation_stroke_sample_weights, validation_stroke_class_weights = utils.calculate_stroke_label_distribution(
        label_user_dict=val_dict,
        swimming_data=swimming_data,
        data_type="validation",
        exclude_label=None
    )

    # Calculate swim style statistics for training and validation sets
    train_counts, train_stats, training_style_sample_weights, training_style_class_weights = utils.calculate_swim_style_distribution(
        label_user_dict=train_dict,
        swimming_data=swimming_data,
        data_type="training",
        exclude_label=None)

    val_counts, val_stats, validation_style_sample_weights, validation_style_class_weights = utils.calculate_swim_style_distribution(
        label_user_dict=val_dict, 
        swimming_data=swimming_data,
        data_type="validation",
        exclude_label=None)

    # Compare distributions between training and validation sets
    if train_stats and val_stats:
        print("\nDistribution Comparison (Training vs Validation):")
        for label in set(train_stats["label_counts"].keys()) | set(val_stats["label_counts"].keys()):
            train_pct = train_stats["distribution"].get(label, 0) * 100
            val_pct = val_stats["distribution"].get(label, 0) * 100
            diff = abs(train_pct - val_pct)
            print(f"  Swim Style {label}: Train {train_pct:.2f}% vs Val {val_pct:.2f}% (Diff: {diff:.2f}%)")

    # The generator used to draw mini-batches
    gen = swimming_data.batch_generator_dicts(train_dict=train_dict,
                                              batch_size=training_parameters['batch_size'],
                                              noise_std=training_parameters['noise_std'],
                                              mirror_prob=training_parameters['mirror_prob'],
                                              random_rot_deg=training_parameters['random_rot_deg'],
                                              use_4D=False,
                                              swim_style_output=training_parameters['swim_style_output'], 
                                              stroke_label_output=training_parameters['stroke_label_output'],
                                              return_stroke_mask=training_parameters['stroke_mask'],
                                              compute_sample_weights=training_parameters['compute_sample_weights'])
                                            #  global_swim_weights=training_style_class_weights,
                                             # global_stroke_weights=training_stroke_class_weights)


    # Optional: Visualize the distributions
    if data_parameters['debug_plot']:
        utils_plot.plot_label_distribution(training_probabilities, validation_probabilities)    
        feature_indices = [0, 1, 2, 3, 4, 5]  # Example: indices of x, y, z axes for accelerometer

        utils_plot.analyze_feature_distribution(train_dict, swimming_data, feature_indices, data_type="training", exclude_label=0)
        utils_plot.analyze_feature_distribution(val_dict, swimming_data, feature_indices, data_type="validation", exclude_label=0)


    if data_parameters['debug']:
        if training_parameters['swim_style_output'] and training_parameters['stroke_label_output']:
            # Retrieve and print a batch for inspection
            batch_data, batch_outputs, batch_sample_weights = next(gen)

            # Debugging combined sensor and stroke data
            print(f"Batch data shape: {batch_data.shape}")

            # Debugging swim style and stroke label outputs
            print(f"Swim Style Output Shape: {batch_outputs['swim_style_output'].shape}")
            print(f"Stroke Label Output Shape: {batch_outputs['stroke_label_output'].shape}")
            print(f"Stroke Label Output Unique Values: {np.unique(batch_outputs['stroke_label_output'])}")

            # Debugging stroke mask
            print(f"Stroke Mask Shape: {batch_sample_weights['stroke_label_output'].shape}")
            print(f"Stroke Mask Unique Values: {np.unique(batch_sample_weights['stroke_label_output'])}")

            # Compare combined stroke data with stroke labels and mask
            example_index = 0  # Choose a batch sample index to debug
            print("\nSample Debug:")
            print(f"Batch Data (Sensor Features + Stroke Labels): {batch_data[example_index, :, :]}")
            print(f"Stroke Labels (from Combined): {batch_outputs['stroke_label_output'][example_index, :, :]}")
            print(f"Stroke Mask (for Combined): {batch_sample_weights['stroke_label_output'][example_index, :, :]}")
        elif training_parameters['swim_style_output']:
            batch_data, batch_outputs = next(gen)

            # Debugging combined sensor and stroke data
            print(f"Batch data shape: {batch_data.shape}")

            # Debugging swim style and stroke label outputs
            print(f"Swim Style Output Shape: {batch_outputs['swim_style_output'].shape}")
            print(f"Swim Style Output Unique Values: {np.unique(batch_outputs['swim_style_output'])}")

            # Compare combined stroke data with stroke labels and mask
            example_index = 0  # Choose a batch sample index to debug
            print("\nSample Debug:")
            print(f"Batch Data (Sensor Features): {batch_data[example_index, :, :]}")
            print(f"Swim Style Labels: {batch_outputs['swim_style_output'][example_index, :]}")
        else:
            # Retrieve and print a batch for inspection
            batch_data, batch_outputs, batch_sample_weights = next(gen)

            # Debugging combined sensor and stroke data
            print(f"Batch data shape: {batch_data.shape}")

            # Debugging swim style and stroke label outputs
            print(f"Stroke Label Output Shape: {batch_outputs['stroke_label_output'].shape}")
            print(f"Stroke Label Output Unique Values: {np.unique(batch_outputs['stroke_label_output'])}")

            # Debugging stroke mask
            print(f"Stroke Mask Shape: {batch_sample_weights['stroke_label_output'].shape}")
            print(f"Stroke Mask Unique Values: {np.unique(batch_sample_weights['stroke_label_output'])}")

            # Compare combined stroke data with stroke labels and mask
            example_index = 0  # Choose a batch sample index to debug
            print("\nSample Debug:")
            print(f"Batch Data (Sensor Features): {batch_data[example_index, :, :]}")
            print(f"Stroke Labels: {batch_outputs['stroke_label_output'][example_index, :, :]}")
            print(f"Stroke Mask: {batch_sample_weights['stroke_label_output'][example_index, :, :]}")
            # Usage:
            # Plot first sample in batch
            utils_plot.plot_batch_and_mask(batch_data, batch_outputs['stroke_label_output'], batch_sample_weights['stroke_label_output'], sample_idx=0)

            # Plot a sample with strokes (if you know the index)
            # Find a sample with strokes
            stroke_samples = np.where(np.any(batch_outputs['stroke_label_output'] == 1, axis=1))[0]
            if len(stroke_samples) > 0:
                print(f"\nPlotting sample with strokes (index {stroke_samples[0]}):")
                utils_plot.plot_batch_and_mask(batch_data, batch_outputs['stroke_label_output'], batch_sample_weights['stroke_label_output'], sample_idx=stroke_samples[0])


    # Optimizer
   # optimizer = tf.keras.optimizers.Adam(learning_rate=training_parameters['lr'], beta_1=training_parameters['beta_1'],
    #                                  beta_2=training_parameters['beta_2'])

    # Get the validation data with weights and mask
    x_val, y_val_sparse, y_val_cat, y_stroke_val, val_sample_weights, val_stroke_mask = swimming_data.get_windows_dict(
        val_dict, return_weights=True, return_mask=True, transition_label=0
    )

    final_swim_weights = val_sample_weights * validation_style_sample_weights
    final_swim_weights = final_swim_weights / np.mean(final_swim_weights)
    final_stroke_weights = val_sample_weights[:, np.newaxis, np.newaxis] * validation_stroke_sample_weights

    model = wavenet_model.build_wavenet_model(input_shape, num_styles=5, nfilt=32, output_bias=training_parameters['output_bias'])
    model.summary()

    model = compile_model(data_parameters, model, training_parameters, loss_parameters=loss_parameters)


    # Print validation data shapes
    if data_parameters['debug']:
        print(f"x_val shape: {x_val.shape}")
        print(f"y_val_cat shape: {y_val_cat.shape}")
        print(f"y_stroke_val shape: {y_stroke_val.shape}")
        print(f"y_val_sparse shape: {y_val_sparse.shape}")
        print(f"val_sample_weights shape: {val_sample_weights.shape}")
        print(f"val_stroke_mask shape: {val_stroke_mask.shape}")
        # Check unique values in stroke mask
        print(f"y_val_cat unique values: {np.unique(y_val_cat)}")
        print(f"y_stroke_val unique values: {np.unique(y_stroke_val)}")
        print(f"y_val_sparse unique values: {np.unique(y_val_sparse)}")
        print(f"val_stroke_mask unique values: {np.unique(val_stroke_mask)}")
        # Compare combined stroke data with stroke labels and mask
        example_index = 0  # Choose a batch sample index to debug
        print("\nSample Debug:")
        print(f"x_val (Sensor Features): {x_val[example_index, :, :]}")
        print(f"y_stroke_val (Stroke Labels): {y_stroke_val[example_index, :, :]}")
        print(f"y_val_cat: {y_val_cat[example_index,  :]}")
        print(f"y_val_sparse: {y_val_sparse[example_index]}")
        print(f"val_stroke_mask: {val_stroke_mask[example_index, :, :]}")
   # batch_data, batch_outputs, batch_sample_weights = next(gen)

    #utils.verify_masking(batch_outputs['stroke_label_output'], batch_sample_weights['stroke_label_output'], "Batch", batch_size=64)
    #utils.verify_masking(y_stroke_val, val_stroke_mask, "Validation", batch_size=64)

    if training_parameters['swim_style_output'] and training_parameters['stroke_label_output']:
        validation_data = (x_val, {'swim_style_output': y_val_cat, 'stroke_label_output': y_stroke_val},
                                {'swim_style_output': validation_style_sample_weights, 'stroke_label_output': (val_stroke_mask 
                                    if training_parameters['stroke_mask'] 
                                    else validation_stroke_sample_weights)})
        # Set up the combined early stopping callback
        #callbacks = utils_train.get_callbacks_regression(experiment_save_path, user_test, log_dir, x_val, y_val_cat, y_stroke_val)
        #callbacks=[utils_train.CombinedAccuracyF1Metrics(x_val, y_val_cat, y_stroke_val, val_sample_weights=validation_stroke_sample_weights)]
        callbacks = [
            EnhancedPeakConsistencyCallback(
                x_val,
                y_stroke_val,
                target_threshold=0.8
            ),
            CombinedMetricCallback(alpha=0.7),
            CombinedEarlyStopping(
                monitor1='val_stroke_label_output_weighted_f1_score',
                monitor2='val_swim_style_output_weighted_categorical_accuracy',
                mode1='max',
                mode2='max',
                patience=30,
                restore_best_weights=True
            ),

           # utils_train.CombinedAccuracyF1Metrics(x_val, y_val_cat, y_stroke_val, val_sample_weights=validation_stroke_sample_weights)
        ]

        callbacks.append(get_callbacks_combined(experiment_save_path, user_test, log_dir))

    elif training_parameters['swim_style_output']:
        validation_data = (x_val, y_val_cat, val_sample_weights)
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_weighted_categorical_accuracy', patience=10, restore_best_weights=True, mode='max')]
        callbacks.append(get_callbacks(experiment_save_path, user_test, log_dir))
    else:
        validation_data = (x_val,  y_stroke_val, (val_stroke_mask if training_parameters['stroke_mask'] else val_stroke_weights))
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_weighted_f1_score', patience=10, restore_best_weights=True, mode='max')]
        callbacks.append(get_callbacks(experiment_save_path, user_test, log_dir))
    
        # Pass to model.fit
    history = model.fit(
        gen,  
        validation_data=validation_data,
        epochs=training_parameters['max_epochs'],
        steps_per_epoch=training_parameters['steps_per_epoch'],
        validation_batch_size=training_parameters['batch_size'],
        callbacks=callbacks,
        verbose=1
    )

    # Save model in h5 format
    model_h5_path = os.path.join(experiment_save_path, f'model_{user_test}.h5')
    model.save(model_h5_path)
    # Save model in native Keras format
    model_keras_path = os.path.join(experiment_save_path, f'model_{user_test}.keras')
    model.save(model_keras_path)
    # Saving the history and parameters
    with open(os.path.join(experiment_save_path, 'train_val_dicts.pkl'), 'wb') as f:
        pickle.dump([train_dict, val_dict], f)
    with open(os.path.join(experiment_save_path, 'history.pkl'), 'wb') as f:
        pickle.dump([history.history], f)
    with open(os.path.join(experiment_save_path, 'data_parameters.pkl'), 'wb') as f:
        pickle.dump([data_parameters], f)

    with open(os.path.join(experiment_save_path, 'training_parameters.pkl'), 'wb') as f:
        pickle.dump([training_parameters], f)

    with open(os.path.join(experiment_save_path, 'loss_parameters.pkl'), 'wb') as f:
        pickle.dump([loss_parameters], f)
