import tensorflow as tf
from utils_train import compile_model, weighted_binary_crossentropy_smooth_class, EarlyStoppingLogger, CombinedEarlyStopping, CombinedMetricCallback
import pickle
import os
import datetime
import json

def get_default_training_parameters():
    """
    Get a default set of parameters used to train a cnn model
    :return: A dictionary containing parameter names and values
    """
    training_parameters = {'swim_style_lr': 0.0005,  # Constant for swim style
                        'stroke_lr': {
                                'initial_lr': 0.0005,
                                'decay_steps': 1000,
                                'decay_rate': 0.9
                            },
                        'beta_1':          0.9,
                        'beta_2':          0.999,
                        'batch_size':      64,
                        'max_epochs':      48,      # Keeping small for quick testing
                        'steps_per_epoch': 100,      # Keeping small for quick testing
                        'noise_std':       0.01,    # Noise standard deviation for data augmentation
                        'mirror_prob':     0.5,     # Probability of reversing a window for data augmentation
                        'random_rot_deg':  30,      # [-30, 30] is the range of rotation degrees we sample for each
                                                    # window in the mini-batch
                        'group_probs':     {'original': 0.7, 'time_scaled_0.9': 0.15, 'time_scaled_1.1': 0.15},
                        'labels':          [0, 1, 2, 3, 4],
                        'stroke_labels': ['stroke_labels'],  # Labels for stroke predictions
                        'stroke_label_output':      True,
                        'swim_style_output':        True,
                        'output_bias':              None
                        }
    return training_parameters

def load_hyperparameters(hyperparameters_path):
    """
    Load hyperparameters from a pickle file.
    
    Args:
        hyperparameters_path: Path to the pickle file containing hyperparameters
        
    Returns:
        Dictionary containing hyperparameters
    """
    with open(hyperparameters_path, 'rb') as f:
        hyperparameters = pickle.load(f)
    
    print(f"Loaded hyperparameters from {hyperparameters_path}")
    return hyperparameters

def create_model_parameters_from_hyperparameters(hyperparameters):
    """
    Convert loaded hyperparameters to model parameter dictionaries for CNN model.
    
    Args:
        hyperparameters: Dictionary of hyperparameters loaded from the pickle file
        
    Returns:
        Dictionaries for swim and stroke model parameters
    """
    # Extract swim model parameters
    swim_model_parameters = {
        'filters': [],
        'kernel_sizes': [],
        'strides': [],
        'max_pooling': [],
        'units': [],
        'activation': [],
        'batch_norm': [],
        'drop_out': [],
        'max_norm': [],
        'l2_reg': [],
        'labels': [0, 1, 2, 3, 4]  # Fixed labels
    }
    
    # Populate swim model parameters
    for i in range(4):  # 4 Conv layers
        if f'swim_filters_{i}' in hyperparameters:
            swim_model_parameters['filters'].append(int(hyperparameters[f'swim_filters_{i}']))
            swim_model_parameters['kernel_sizes'].append(int(hyperparameters[f'swim_kernel_size_{i}']))
            swim_model_parameters['strides'].append(int(hyperparameters[f'swim_stride_{i}']))
            swim_model_parameters['max_pooling'].append(int(hyperparameters[f'swim_max_pooling_{i}']))
    
    for i in range(1):  # 1 Dense layer
        if f'swim_dense_units_{i}' in hyperparameters:
            swim_model_parameters['units'].append(int(hyperparameters[f'swim_dense_units_{i}']))
    
    for i in range(5):  # 5 activation/batch_norm/dropout configurations (4 conv + 1 dense)
        if f'swim_activation_{i}' in hyperparameters:
            swim_model_parameters['activation'].append(hyperparameters[f'swim_activation_{i}'])
            swim_model_parameters['batch_norm'].append(hyperparameters[f'swim_batch_norm_{i}'])
            swim_model_parameters['drop_out'].append(hyperparameters[f'swim_dropout_{i}'])
            swim_model_parameters['max_norm'].append(hyperparameters[f'swim_max_norm_{i}'])
            swim_model_parameters['l2_reg'].append(hyperparameters[f'swim_l2_reg_{i}'])
    
    # Extract stroke model parameters
    stroke_model_parameters = {
        'filters': [],
        'kernel_sizes': [],
        'strides': [],
        'max_pooling': [],
        'units': [],
        'lstm_dropout': 0.0,
        'lstm_recurrent_dropout': 0.0,
        'lstm_l2_reg': 0.0,
        'lstm_max_norm': 0.0,
        'activation': [],
        'batch_norm': [],
        'drop_out': [],
        'max_norm': [],
        'l2_reg': [],
        'final_dropout': 0.0,
        'stroke_labels': ['stroke_labels']
    }
    
    # Populate stroke model parameters
    for i in range(2):  # 2 Conv layers
        if f'stroke_filters_{i}' in hyperparameters:
            stroke_model_parameters['filters'].append(int(hyperparameters[f'stroke_filters_{i}']))
            stroke_model_parameters['kernel_sizes'].append(int(hyperparameters[f'stroke_kernel_size_{i}']))
            stroke_model_parameters['strides'].append(int(hyperparameters[f'stroke_stride_{i}']))
            stroke_model_parameters['max_pooling'].append(int(hyperparameters[f'stroke_max_pooling_{i}']))
    
    # LSTM parameters
    if 'stroke_lstm_units' in hyperparameters:
        stroke_model_parameters['units'].append(int(hyperparameters['stroke_lstm_units']))
        stroke_model_parameters['lstm_dropout'] = hyperparameters['stroke_lstm_dropout']
        stroke_model_parameters['lstm_recurrent_dropout'] = hyperparameters['stroke_lstm_recurrent_dropout']
        stroke_model_parameters['lstm_l2_reg'] = hyperparameters['stroke_lstm_l2_reg']
        stroke_model_parameters['lstm_max_norm'] = hyperparameters['stroke_lstm_max_norm']
    
    for i in range(2):  # 2 activation functions for conv layers
        if f'stroke_activation_{i}' in hyperparameters:
            stroke_model_parameters['activation'].append(hyperparameters[f'stroke_activation_{i}'])
    
    for i in range(3):  # 3 sets of regularization params (2 conv + 1 extra)
        if f'stroke_batch_norm_{i}' in hyperparameters:
            stroke_model_parameters['batch_norm'].append(hyperparameters[f'stroke_batch_norm_{i}'])
            stroke_model_parameters['drop_out'].append(hyperparameters[f'stroke_dropout_{i}'])
            stroke_model_parameters['max_norm'].append(hyperparameters[f'stroke_max_norm_{i}'])
            stroke_model_parameters['l2_reg'].append(hyperparameters[f'stroke_l2_reg_{i}'])
    
    if 'stroke_final_dropout' in hyperparameters:
        stroke_model_parameters['final_dropout'] = hyperparameters['stroke_final_dropout']
    
    # Extract learning rates
    swim_style_lr = hyperparameters.get('swim_style_lr')
    stroke_lr = hyperparameters.get('stroke_lr')
    
    return swim_model_parameters, stroke_model_parameters, swim_style_lr, stroke_lr

def cnn_model(input_shape, swim_model_parameters=None, stroke_model_parameters=None, training_parameters=None):

    inputs = tf.keras.Input(shape=input_shape)

    # Build swim style branch if swim_model_parameters are provided
    swim_style_output = None
    if swim_model_parameters is not None and training_parameters['swim_style_output']:
        swim_style_output = swim_style_model(inputs, swim_model_parameters, output_bias=None)

    # Build stroke branch if stroke_model_parameters are provided
    stroke_label_output = None
    if stroke_model_parameters is not None and training_parameters['stroke_label_output']:
        stroke_label_output = stroke_model(inputs, stroke_model_parameters, output_bias=training_parameters['output_bias'])

    # Combine outputs based on the branches enabled
    if swim_style_output is not None and stroke_label_output is not None:
        model = tf.keras.Model(inputs=inputs, outputs=[swim_style_output, stroke_label_output])
    elif swim_style_output is not None:
        model = tf.keras.Model(inputs=inputs, outputs=swim_style_output)
    elif stroke_label_output is not None:
        model = tf.keras.Model(inputs=inputs, outputs=stroke_label_output)
    else:
        raise ValueError("No outputs selected for the model.")

    return model

def swim_style_model(inputs, swim_model_parameters, use_seed=True, output_bias=None):
    """
    Create a CNN model for swim style classification
    :param input_shape: The shape of the input data
    :param swim_model_parameters: A dictionary containing the parameters for the model
    :return: A tf.keras.Model object
    """
    num_cl = len(swim_model_parameters['filters'])
    num_fcl = len(swim_model_parameters['units'])
    cnt_layer = 0

    swim_style_branch = inputs
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    # Main convolutional layers (for swim style)
    for i in range(num_cl):
        kernel_constraint = (
            tf.keras.constraints.max_norm(swim_model_parameters['max_norm'][cnt_layer])
            if swim_model_parameters['max_norm'][cnt_layer] != 0
            else None
        )
        kernel_regularizer = (
            tf.keras.regularizers.l2(swim_model_parameters['l2_reg'][cnt_layer])
            if swim_model_parameters['l2_reg'][cnt_layer] != 0
            else None
        )
        strides = swim_model_parameters['strides'][i]
        strides = 1 if strides == 0 else (strides, 1)

        swim_style_branch = tf.keras.layers.Conv2D(
            filters=swim_model_parameters['filters'][i],
            kernel_size=(swim_model_parameters['kernel_sizes'][i], 1),
            strides=strides,
            padding='same',  # Use 'same' padding
            kernel_constraint=kernel_constraint,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=tf.keras.initializers.he_normal(seed=use_seed and 1337),
            bias_initializer="zeros",
            name=f'swim_style_conv_{i}'
        )(swim_style_branch)

        if swim_model_parameters['batch_norm'][cnt_layer]:
            swim_style_branch = tf.keras.layers.BatchNormalization(
                name=f'swim_style_bn_{i}'
            )(swim_style_branch)
        # Handle different activation functions
        activation_function = swim_model_parameters['activation'][cnt_layer]
        if activation_function == 'leaky_relu':
            swim_style_branch = tf.keras.layers.LeakyReLU(alpha=0.2, name=f'swim_style_activation_{i}')(swim_style_branch)
        else:
            swim_style_branch = tf.keras.layers.Activation(activation_function, name=f'swim_style_activation_{i}')(swim_style_branch)
        max_pooling = swim_model_parameters['max_pooling'][i]
        if max_pooling != 0:
            swim_style_branch = tf.keras.layers.MaxPooling2D(
                (max_pooling, 1),
                name=f'swim_style_pool_{i}'
            )(swim_style_branch)
        if swim_model_parameters['drop_out'][cnt_layer] is not None:
            swim_style_branch = tf.keras.layers.Dropout(
                swim_model_parameters['drop_out'][cnt_layer],
                seed=use_seed and 1337,
                name=f'swim_style_dropout_{i}'
            )(swim_style_branch)
        cnt_layer += 1

    # Swim Style Branch
    swim_style_branch = tf.keras.layers.Flatten(
        name='swim_style_flatten'
    )(swim_style_branch)
    
    for i in range(num_fcl):
        kernel_constraint = (
            tf.keras.constraints.max_norm(swim_model_parameters['max_norm'][cnt_layer])
            if swim_model_parameters['max_norm'][cnt_layer] != 0
            else None
        )
        kernel_regularizer = (
            tf.keras.regularizers.l2(swim_model_parameters['l2_reg'][cnt_layer])
            if swim_model_parameters['l2_reg'][cnt_layer] != 0
            else None
        )
        swim_style_branch = tf.keras.layers.Dense(
            units=swim_model_parameters['units'][i],
            kernel_constraint=kernel_constraint,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=tf.keras.initializers.he_uniform(seed=use_seed and 1337),
            bias_initializer='zeros',
            name=f'swim_style_dense_{i}'
        )(swim_style_branch)
        
        if swim_model_parameters['batch_norm'][cnt_layer]:
            swim_style_branch = tf.keras.layers.BatchNormalization(
                name=f'swim_style_dense_bn_{i}'
            )(swim_style_branch)
        swim_style_branch = tf.keras.layers.Activation(
            swim_model_parameters['activation'][cnt_layer],
            name=f'swim_style_dense_activation_{i}'
        )(swim_style_branch)
        if swim_model_parameters['drop_out'][cnt_layer] is not None:
            swim_style_branch = tf.keras.layers.Dropout(
                swim_model_parameters['drop_out'][cnt_layer],
                seed=use_seed and 1337,
                name=f'swim_style_dense_dropout_{i}'
            )(swim_style_branch)
        cnt_layer += 1

    # Swim Style Output
    swim_style_output = tf.keras.layers.Dense(
        len(swim_model_parameters['labels']),
        activation="softmax",
        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=use_seed and 1337),
        name="swim_style_output"
    )(swim_style_branch)

    return swim_style_output

def stroke_model(inputs, stroke_model_parameters, use_seed=True, output_bias=None):
    """
    Create a CNN model for stroke detection
    :param input_shape: The shape of the input data
    :param stroke_model_parameters: A dictionary containing the parameters for the model
    :return: A tf.keras.Model object
    """
    # Stroke Detection Branch
    num_stroke_cl = len(stroke_model_parameters['filters'])
    cnt_layer = 0
    stroke_branch = inputs
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    # Convolutional layers for stroke detection
    for i in range(num_stroke_cl):
        kernel_constraint = (
            tf.keras.constraints.max_norm(stroke_model_parameters['max_norm'][cnt_layer])
            if stroke_model_parameters['max_norm'][cnt_layer] != 0
            else None
        )
        kernel_regularizer = (
            tf.keras.regularizers.l2(stroke_model_parameters['l2_reg'][cnt_layer])
            if stroke_model_parameters['l2_reg'][cnt_layer] != 0
            else None
        )

        kernel_initializer=tf.keras.initializers.he_normal(seed=use_seed and 1337)

        strides = stroke_model_parameters['strides'][i]
        strides = 1 if strides == 0 else (strides, 1)

        stroke_branch = tf.keras.layers.Conv2D(
            filters=stroke_model_parameters['filters'][i],
            kernel_size=(stroke_model_parameters['kernel_sizes'][i], 1),
            strides=strides,
            padding='same',
            kernel_constraint=kernel_constraint,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer,
            bias_initializer="zeros",
            name=f'stroke_conv_{i}'
        )(stroke_branch)

        if stroke_model_parameters['batch_norm'][cnt_layer]:
            stroke_branch = tf.keras.layers.BatchNormalization(
                name=f'stroke_bn_{i}'
            )(stroke_branch)
        stroke_branch = tf.keras.layers.Activation(
            stroke_model_parameters['activation'][cnt_layer],
            name=f'stroke_activation_{i}'
        )(stroke_branch)
        max_pooling = stroke_model_parameters['max_pooling'][i]
        if max_pooling != 0:
            stroke_branch = tf.keras.layers.MaxPooling2D(
                (max_pooling, 1),
                name=f'stroke_pool_{i}'
            )(stroke_branch)
        if stroke_model_parameters['drop_out'][cnt_layer] is not None:
            stroke_branch = tf.keras.layers.Dropout(
                stroke_model_parameters['drop_out'][cnt_layer],
                seed=use_seed and 1337,
                name=f'stroke_dropout_{i}'
            )(stroke_branch)
        
        # If this is the first conv layer, create attention mechanism
        if i == 0:
            attention = stroke_branch
        cnt_layer += 1

    # Apply global pooling to the attention tensor
    attention = tf.keras.layers.GlobalAveragePooling2D(name='stroke_attention_global_pooling')(attention)

    # Expand dimensions of the attention tensor to match the feature tensor
    attention = tf.keras.layers.Dense(
        units=stroke_branch.shape[-1],  # Match the number of filters in the feature tensor
        activation='sigmoid',  # Use sigmoid to scale attention values between 0 and 1
        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=use_seed and 1337),
        name='stroke_attention_dense'
    )(attention)
    attention = tf.keras.layers.Reshape((1, 1, stroke_branch.shape[-1]), name='stroke_attention_reshape')(attention)

    # Multiply attention with features
    stroke_branch = tf.keras.layers.Multiply(name='stroke_multiply')([stroke_branch, attention])

    
    # Reshape to (batch, time, features)
    temporal_dim = stroke_branch.shape[1]
    stroke_branch = tf.keras.layers.Reshape(
        (temporal_dim, -1),
        name='stroke_reshape'
    )(stroke_branch)
    
    # LSTM layer
    stroke_branch = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            units=stroke_model_parameters['units'][0],  # Tunable LSTM units
            return_sequences=True,
            dropout=stroke_model_parameters['lstm_dropout'],  # Tunable LSTM dropout
            recurrent_dropout=stroke_model_parameters['lstm_recurrent_dropout'],  # Tunable recurrent dropout
            recurrent_initializer='orthogonal',  # Good for LSTM
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=use_seed and 1337),
            kernel_constraint=tf.keras.constraints.max_norm(stroke_model_parameters['lstm_max_norm'])  # Tunable max norm
                if stroke_model_parameters['lstm_max_norm'] != 0
                else None,
            kernel_regularizer=tf.keras.regularizers.l2(stroke_model_parameters['lstm_l2_reg'])  # Tunable L2 regularization
                if stroke_model_parameters['lstm_l2_reg'] != 0
                else None,
            bias_initializer='zeros',
            name='stroke_lstm'
        ),
        merge_mode='concat',
        name='stroke_bilstm'
    )(stroke_branch)

    if stroke_model_parameters['final_dropout'] > 0.0:
        stroke_branch = tf.keras.layers.Dropout(
            stroke_model_parameters['final_dropout'],
            name='stroke_final_dropout'
        )(stroke_branch)

    stroke_branch = tf.keras.layers.LayerNormalization(
        name='stroke_layer_norm'
    )(stroke_branch)

    # Stroke detection output
    stroke_label_output = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(
            len(stroke_model_parameters['stroke_labels']),
            activation="sigmoid",
            bias_initializer=output_bias,
            kernel_initializer=tf.keras.initializers.glorot_normal(seed=use_seed and 1337),
            name='stroke_dense'
        ),
        name="stroke_label_output"
    )(stroke_branch)

    return stroke_label_output

class DebugCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch + 1} Logs:")
        for key, value in logs.items():
            print(f"{key}: {value}")

def get_tensorboard_callback(log_dir):
    """
    Create a TensorBoard callback for logging hyperparameters and metrics.
    """
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,  # Log histograms of weights every epoch
        write_graph=True,  # Log the computation graph
        write_images=True,  # Log model weights as images
        update_freq='epoch',  # Log metrics at the end of each epoch
        profile_batch=0  # Disable profiling to avoid overhead
    )
    return tensorboard_callback

def train_with_best_hyperparameters(hyperparameters_path, input_shape, training_parameters):
    """
    Train a CNN model using hyperparameters loaded from a pickle file.
    
    Args:
        hyperparameters_path: Path to the pickle file containing hyperparameters
        input_shape: Shape of the input data
        data_parameters: Dictionary of data parameters
        training_parameters: Dictionary of training parameters
        class_weights: Weights for each class
        train_gen: Generator for training data
        validation_data: Validation data
        run_name: Name for this training run (for logs)
        callbacks: List of callbacks for training
        
    Returns:
        Trained model and training history
    """
    # Load hyperparameters
    hyperparameters = load_hyperparameters(hyperparameters_path)
    
    # Convert hyperparameters to model parameters
    swim_model_parameters, stroke_model_parameters, swim_style_lr, stroke_lr = create_model_parameters_from_hyperparameters(hyperparameters)
    
    # Print model parameters for debugging
    print("\nSwim model parameters:")
    print(json.dumps({k: v for k, v in swim_model_parameters.items() if k != 'labels'}, indent=2, default=str))
    
    print("\nStroke model parameters:")
    print(json.dumps({k: v for k, v in stroke_model_parameters.items() if k != 'stroke_labels'}, indent=2, default=str))
    
    # Update learning rates in training parameters if available
    if swim_style_lr is not None:
        print(f"Using swim style learning rate from hyperparameters: {swim_style_lr}")
        training_parameters['swim_style_lr'] = swim_style_lr
    
    if stroke_lr is not None:
        print(f"Using stroke learning rate from hyperparameters: {stroke_lr}")
        if isinstance(training_parameters['stroke_lr'], dict):
            # If stroke_lr is expected to be a dict with decay settings
            training_parameters['stroke_lr']['initial_lr'] = stroke_lr
        else:
            training_parameters['stroke_lr'] = stroke_lr
    
    # Create the model
    model = cnn_model(
        input_shape=input_shape,
        swim_model_parameters=swim_model_parameters if training_parameters['swim_style_output'] else None,
        stroke_model_parameters=stroke_model_parameters if training_parameters['stroke_label_output'] else None,
        training_parameters=training_parameters
    )
    
    return model

if __name__ == '__main__':
    # Set up basic parameters
    input_shape = (180, 6, 1)  # CNN expects a different input shape than LSTM
    data_parameters = {'label_type': 'majority', 'debug': True}
    training_parameters = get_default_training_parameters()
    
    # Path to the best hyperparameters pickle file
    run_name = "tune_swim_stroke_100_weighted"
    hyperparameters_path = f'best_hyperparameters/{run_name}/best_hyperparameters_1.pkl'
    
    # Class weights
    class_weights = [1.0, 15.0]  # Example class weights for strokes
    
    # Configure generators and callbacks based on model outputs
    if training_parameters['swim_style_output'] and training_parameters['stroke_label_output']:
        # Combined output model (both swim style and stroke detection)
        def gen():
            while True:
                yield (
                    tf.random.normal((64, 180, 6, 1)),  # Features with channel dimension
                    {  # Labels
                        'swim_style_output': tf.random.uniform((64, 5), minval=0, maxval=5, dtype=tf.int32),
                        'stroke_label_output': tf.random.uniform((64, 180, 1), minval=0, maxval=2, dtype=tf.int32),
                    }
                )

        def val_gen():
            while True:
                yield (
                    tf.random.normal((64, 180, 6, 1)),  # Features with channel dimension
                    { 
                        'swim_style_output': tf.random.uniform((64, 5), minval=0, maxval=5, dtype=tf.int32),
                        'stroke_label_output': tf.random.uniform((64, 180, 1), minval=0, maxval=2, dtype=tf.int32),
                    }
                )
        
        # Set up the combined callbacks
        callbacks = [
            CombinedMetricCallback(alpha=0.5),
            CombinedEarlyStopping(
                monitor1='val_stroke_label_output_weighted_f1_score',
                monitor2='val_swim_style_output_weighted_categorical_accuracy',
                mode1='max',
                mode2='max',
                patience=10,
                restore_best_weights=True
            )
        ]
        
    elif training_parameters['swim_style_output']:  
        # Only swim style output
        def gen():
            while True:
                yield (
                    tf.random.normal((64, 180, 6, 1)),  # Features
                    tf.random.uniform((64, 5), minval=0, maxval=5, dtype=tf.int32)  # Labels
                )
                
        def val_gen():
            while True:
                yield (
                    tf.random.normal((64, 180, 6, 1)),  # Features
                    tf.random.uniform((64, 5), minval=0, maxval=5, dtype=tf.int32)  # Labels
                )

        callbacks = [tf.keras.callbacks.EarlyStopping(
            monitor='val_weighted_categorical_accuracy', 
            patience=10, 
            restore_best_weights=True, 
            mode='max'
        )]
        
    else:  
        # Only stroke label output
        def gen():
            while True:
                yield (
                    tf.random.normal((64, 180, 6, 1)),  # Features
                    tf.random.uniform((64, 180, 1), minval=0, maxval=2, dtype=tf.int32)  # Labels
                )
                
        def val_gen():
            while True:
                yield (
                    tf.random.normal((64, 180, 6, 1)),  # Features
                    tf.random.uniform((64, 180, 1), minval=0, maxval=2, dtype=tf.int32)  # Labels
                )
                
        callbacks = [tf.keras.callbacks.EarlyStopping(
            monitor='val_weighted_f1_score', 
            patience=10, 
            restore_best_weights=True, 
            mode='max'
        )]
    
    # Train model with the best hyperparameters
    print(f"Training CNN model with best hyperparameters from: {hyperparameters_path}")
    model = train_with_best_hyperparameters(
        hyperparameters_path=hyperparameters_path,
        input_shape=input_shape,
    #    data_parameters=data_parameters,
        training_parameters=training_parameters,
     #   class_weights=class_weights,
      #  train_gen=gen(),
       # validation_data=val_gen(),
      #  run_name=f"training_{run_name}",
       # callbacks=callbacks
    )
    
    # Print model summary
    print("\nFinal model summary:")
    model.summary()
    
    # Save the trained model
    model_save_path = f"models/{run_name}_best_model"
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
