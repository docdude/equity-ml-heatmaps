import tensorflow as tf
import os
import tensorflow_addons as tfa
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

class ManualWeightedHuber(tf.keras.losses.Loss):
    def __init__(self, delta=1.0, name="manual_weighted_huber"):
        super().__init__(name=name)
        self.delta = delta

    def call(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, dtype=tf.float32)
        error = y_true - y_pred
        abs_error = tf.abs(error)

        # Huber loss calculation
        quadratic = tf.minimum(abs_error, self.delta)
        linear = abs_error - quadratic
        loss = 0.5 * tf.square(quadratic) + self.delta * linear  # shape: (batch, 180, 1)

        tf.print(">>> Huber loss shape:", tf.shape(loss))
        if sample_weight is not None:
            tf.print(">>> Sample weight shape:", tf.shape(sample_weight))
            sample_weight = tf.cast(sample_weight, loss.dtype)
            loss *= sample_weight  # shape still (batch, 180, 1)

        # Reduce mean over all time steps and batch
        return tf.reduce_mean(loss)

def weighted_binary_crossentropy(weight_zero=1.0, weight_one=15.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # Reshape y_true to match y_pred if needed
        y_true = tf.reshape(y_true, tf.shape(y_pred))
        
        # Calculate weights maintaining the shape
        weights = y_true * weight_one + (1 - y_true) * weight_zero
        
        # Calculate binary crossentropy
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        
        # Ensure shapes match for multiplication
        weights = tf.squeeze(weights, axis=-1)
        
        # Calculate weighted loss
        weighted_loss = weights * bce
        return tf.reduce_mean(weighted_loss)
    return loss

 
def weighted_binary_crossentropy_smooth_class(class_weights):  
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_true = tf.reshape(y_true, tf.shape(y_pred))
        
        # Use calculated class weights
        weight_zero, weight_one = class_weights
        weights = y_true * weight_one + (1 - y_true) * weight_zero
        weights = tf.squeeze(weights, axis=-1)

        # BCE with label smoothing
        bce = tf.keras.losses.binary_crossentropy(
            y_true, 
            y_pred, 
            label_smoothing=0.1  # Smooth labels directly
        )
        
        return tf.reduce_mean(weights * bce)
    return loss

def focal_loss(gamma=2.0, alpha=0.75):
    def focal_loss_fn(y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        p_t = tf.clip_by_value(p_t, 1e-7, 1.0 - 1e-7)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        focal_loss = -alpha_t * (1 - p_t) ** gamma * tf.math.log(p_t)
        
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            while len(sample_weight.shape) < len(focal_loss.shape):
                sample_weight = tf.expand_dims(sample_weight, -1)
            focal_loss = focal_loss * sample_weight
        
        focal_loss = tf.reduce_mean(focal_loss, axis=[1, 2])  # mean over timesteps and features
        return tf.reduce_mean(focal_loss)  # mean over batch
    return focal_loss_fn

def wrapped_huber_loss(delta=1.0):
    base_loss = tf.keras.losses.Huber(delta=delta, reduction=tf.keras.losses.Reduction.NONE)
    
    def loss_fn(y_true, y_pred):
        return tf.reduce_mean(base_loss(y_true, y_pred))  # Don't apply weights here!
    
    return loss_fn

class CustomBFCEWithPeakConsistency(tf.keras.losses.Loss):
    def __init__(self, alpha=0.85, gamma=1.0, peak_weight=0.1, target_height=0.85):
        super().__init__()
        self.bfce = tf.keras.losses.BinaryFocalCrossentropy(
            alpha=alpha, gamma=gamma, from_logits=False
        )
        self.peak_weight = peak_weight
        self.target_height = target_height
    
    def call(self, y_true, y_pred, sample_weight=None):
        bfce_loss = self.bfce(y_true, y_pred, sample_weight)
        
        peak_mask = tf.cast(y_true > 0.7, tf.float32)
        peak_preds = y_pred * peak_mask
        peak_loss = tf.square(peak_preds - self.target_height * peak_mask)
        
        if sample_weight is not None:
            tf.print(">>> Sample weight shape:", tf.shape(sample_weight))
            sample_weight = tf.cast(sample_weight, loss.dtype)
            peak_loss = peak_loss * sample_weight
            
        peak_loss = tf.reduce_mean(peak_loss)
        return bfce_loss + self.peak_weight * peak_loss

class ConsistentPeakBFCE(tf.keras.losses.Loss):
    def __init__(self, 
                 alpha=0.85, 
                 gamma=1.25, 
                 target_height=0.85, 
                 peak_weight=0.1,
                 peak_threshold=0.6,  # Changed from 0.7 to capture more of sigma=2.0 Gaussian
                 name="consistent_peak_bfce"):
        super().__init__(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.target_height = target_height
        self.peak_weight = peak_weight
        self.peak_threshold = peak_threshold
        
        # Create the base BFCE loss
        self.bfce = tf.keras.losses.BinaryFocalCrossentropy(
            alpha=self.alpha,
            gamma=self.gamma,
            from_logits=False,
            reduction=tf.keras.losses.Reduction.NONE
        )
    
    def call(self, y_true, y_pred, sample_weight=None):
        # Cast to float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Compute focal loss
        focal_loss = self.bfce(y_true, y_pred)  # Shape: (B, T)
        
        # Apply sample weights to focal loss if provided
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            sample_weight = tf.squeeze(sample_weight, -1)  # Remove last dim if (B,T,1)
            focal_loss = focal_loss * sample_weight
        
        # Peak consistency loss - only penalize LOW predictions in peak regions
        peak_regions = tf.cast(y_true > self.peak_threshold, tf.float32)  # Where we expect peaks
        low_predictions = tf.cast(y_pred < self.target_height, tf.float32)  # Predictions below target
        penalty_mask = peak_regions * low_predictions  # Only penalize low predictions in peak regions
        
        # Calculate penalty only for low peaks
        target_values = self.target_height * penalty_mask
        actual_values = y_pred * penalty_mask
        peak_mse = tf.square(actual_values - target_values)
        
        # Apply sample weights to peak loss if provided
        if sample_weight is not None:
            peak_mse = peak_mse * sample_weight
        
        # Compute mean peak loss only where penalties apply
        num_penalty_points = tf.reduce_sum(penalty_mask)
        if tf.greater(num_penalty_points, 0):
            peak_loss = tf.reduce_sum(peak_mse) / num_penalty_points
        else:
            peak_loss = 0.0
        
        # Combine losses
        total_loss = focal_loss + self.peak_weight * peak_loss
        
        # Return per-sample loss for SUM_OVER_BATCH_SIZE reduction
        return tf.reduce_mean(total_loss, axis=-1)  # Shape: (B,)



class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._current_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch
        if logs is None:
            logs = {}
        
        # Get learning rate
        optimizer = self.model.optimizer
        if hasattr(optimizer, 'optimizers_and_layers'):
            # Multi-optimizer case
            for i, (opt, _) in enumerate(optimizer.optimizers_and_layers):
                if isinstance(opt.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
                    lr = opt.learning_rate(opt.iterations)
                else:
                    lr = opt.learning_rate
                lr_value = float(lr)
                logs[f'lr_optimizer_{i}'] = lr_value
                
                # Write to TensorBoard
                tf.summary.scalar(f'learning_rate_optimizer_{i}', data=lr_value, step=epoch)
        else:
            # Single optimizer case
            if isinstance(optimizer.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
                lr = optimizer.learning_rate(optimizer.iterations)
            else:
                lr = optimizer.learning_rate
            lr_value = float(lr)
            logs['lr'] = lr_value
            
            # Write to TensorBoard
            tf.summary.scalar('learning_rate', data=lr_value, step=epoch)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        
        # Ensure learning rate is in logs for CSV logger
        optimizer = self.model.optimizer
        if hasattr(optimizer, 'optimizers_and_layers'):
            for i, (opt, _) in enumerate(optimizer.optimizers_and_layers):
                if isinstance(opt.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
                    lr = opt.learning_rate(opt.iterations)
                else:
                    lr = opt.learning_rate
                logs[f'lr_optimizer_{i}'] = float(lr)
        else:
            if isinstance(optimizer.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
                lr = optimizer.learning_rate(optimizer.iterations)
            else:
                lr = optimizer.learning_rate
            logs['lr'] = float(lr)
            
class MultiOptimizerLRReducer(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', factor=0.5, patience=5, 
                 verbose=1, mode='auto', min_lr=1e-6, min_delta=1e-4):
        super(MultiOptimizerLRReducer, self).__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.min_lr = min_lr
        self.min_delta = min_delta
        
        # Initialize metrics
        self.best = float('inf') if mode == 'min' else -float('inf')
        self.wait = 0
        
        # Decide whether to maximize or minimize
        if mode == 'min':
            self.monitor_op = lambda a, b: a < b - min_delta
        elif mode == 'max':
            self.monitor_op = lambda a, b: a > b + min_delta
        else:  # Auto mode
            if 'acc' in self.monitor or 'f1' in self.monitor:
                self.mode = 'max'
                self.monitor_op = lambda a, b: a > b + min_delta
            else:
                self.mode = 'min'
                self.monitor_op = lambda a, b: a < b - min_delta
            
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            print(f"Warning: {self.monitor} not available in logs")
            return
            
        if self.monitor_op(current, self.best):
            # Improvement - reset wait counter and update best
            self.best = current
            self.wait = 0
        else:
            # No improvement
            self.wait += 1
            if self.wait >= self.patience:
                # Time to reduce LR for all optimizers in the multi-optimizer
                if self.verbose:
                    print(f"\nEpoch {epoch}: reducing learning rates.")
                
                optimizer = self.model.optimizer
                
                # Check if using TFA MultiOptimizer
                if isinstance(optimizer, tfa.optimizers.MultiOptimizer):
                    for i, spec in enumerate(optimizer.optimizer_specs):
                        # Get the optimizer from the spec dictionary
                        opt = spec['optimizer']
                        
                        # Get current learning rate
                        current_lr = float(tf.keras.backend.get_value(opt.learning_rate))
                        new_lr = max(current_lr * self.factor, self.min_lr)
                        
                        if self.verbose:
                            print(f"Optimizer {i+1}: {current_lr:.6f} -> {new_lr:.6f}")
                        
                        # Set the new learning rate
                        tf.keras.backend.set_value(opt.learning_rate, new_lr)
                
                # Handle legacy or other MultiOptimizer implementations
                elif hasattr(optimizer, '_optimizer_variables_and_layers'):
                    for i, (opt, _) in enumerate(optimizer._optimizer_variables_and_layers):
                        current_lr = float(tf.keras.backend.get_value(opt.learning_rate))
                        new_lr = max(current_lr * self.factor, self.min_lr)
                        
                        if self.verbose:
                            print(f"Optimizer {i+1}: {current_lr:.6f} -> {new_lr:.6f}")
                        
                        # Set the new learning rate
                        tf.keras.backend.set_value(opt.learning_rate, new_lr)
                
                # Handle single optimizer case
                else:
                    current_lr = float(tf.keras.backend.get_value(optimizer.learning_rate))
                    new_lr = max(current_lr * self.factor, self.min_lr)
                    
                    if self.verbose:
                        print(f"Learning rate: {current_lr:.6f} -> {new_lr:.6f}")
                    
                    # Set the new learning rate
                    tf.keras.backend.set_value(optimizer.learning_rate, new_lr)
                
                # Reset wait counter after reducing LR
                self.wait = 0

class TaskSpecificLRReducer(tf.keras.callbacks.Callback):
    def __init__(self, monitor_settings, verbose=1):
        """
        Task-specific learning rate reducer.
        
        Args:
            monitor_settings: List of tuples (optimizer_index, metric_name, mode, factor, patience, min_lr)
                optimizer_index: Index of the optimizer in MultiOptimizer (0 for common, 1 for swim, 2 for stroke)
                metric_name: Name of metric to monitor
                mode: 'min' or 'max'
                factor: Factor to reduce learning rate by
                patience: How many epochs to wait before reducing
                min_lr: Minimum learning rate
            verbose: Verbosity mode (0=silent, 1=normal, 2=detailed)
        """
        super(TaskSpecificLRReducer, self).__init__()
        self.monitor_settings = monitor_settings
        self.waiting = [0] * len(monitor_settings)
        self.best_values = []
        self.verbose = verbose
        
        # Initialize best values based on mode
        for _, _, mode, _, _, _ in monitor_settings:
            self.best_values.append(float('inf') if mode == 'min' else -float('inf'))
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        optimizer = self.model.optimizer
        
        if not isinstance(optimizer, tfa.optimizers.MultiOptimizer):
            if self.verbose > 0:
                print("Warning: This callback requires a TFA MultiOptimizer")
            return
            
        # Check each monitored metric
        for i, (opt_idx, metric, mode, factor, patience, min_lr) in enumerate(self.monitor_settings):
            current = logs.get(metric)
            
            if current is None:
                if self.verbose > 0:
                    print(f"Warning: {metric} not available in logs")
                continue
                
            # Check if improved
            improved = (mode == 'min' and current < self.best_values[i]) or \
                      (mode == 'max' and current > self.best_values[i])
                
            if improved:
                self.best_values[i] = current
                self.waiting[i] = 0
                if self.verbose > 1:
                    print(f"Optimizer {opt_idx} ({metric}): improved to {current:.4f}")
            else:
                self.waiting[i] += 1
                if self.verbose > 1:
                    print(f"Optimizer {opt_idx} ({metric}): no improvement for {self.waiting[i]}/{patience} epochs. Current: {current:.4f}, Best: {self.best_values[i]:.4f}")
                
                if self.waiting[i] >= patience:
                    # Time to reduce this specific learning rate
                    if opt_idx < len(optimizer.optimizer_specs):
                        opt = optimizer.optimizer_specs[opt_idx]['optimizer']
                        current_lr = float(tf.keras.backend.get_value(opt.learning_rate))
                        new_lr = max(current_lr * factor, min_lr)
                        
                        if self.verbose > 0:
                            print(f"\nEpoch {epoch}: reducing learning rate for optimizer {opt_idx}")
                            print(f"Monitored metric: {metric} = {current:.4f}")
                            print(f"Learning rate: {current_lr:.6f} -> {new_lr:.6f}")
                        
                        tf.keras.backend.set_value(opt.learning_rate, new_lr)
                        self.waiting[i] = 0

class LearningRateLogger_new(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, **kwargs):
        super().__init__(**kwargs)
        self.log_dir = log_dir
        self._current_epoch = 0
        self.writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        optimizer = self.model.optimizer

        # Handle TFA MultiOptimizer case
        if isinstance(optimizer, tfa.optimizers.MultiOptimizer):
            for i, spec in enumerate(optimizer.optimizer_specs):
                # Extract the optimizer using the 'optimizer' key
                opt = spec['optimizer']
                    
                if isinstance(opt.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
                    lr = opt.learning_rate(opt.iterations)
                else:
                    lr = opt.learning_rate
                
                # Get numerical value of learning rate
                lr_value = float(tf.keras.backend.get_value(lr))

                # Log to console for debugging
                print(f"Optimizer {i} learning rate: {lr_value:.8f}")
                logs[f'lr_optimizer_{i}'] = float(lr_value)

                # Write custom scalar for TensorBoard
                with self.writer.as_default():
                    tf.summary.scalar(f'learning_rate_optimizer_{i}', data=lr_value, step=epoch)
                   # self.writer.flush()
                    
        # Handle legacy MultiOptimizer case
        elif hasattr(optimizer, 'optimizers_and_layers'):
            for i, (opt, _) in enumerate(optimizer.optimizers_and_layers):
                if isinstance(opt.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
                    lr = opt.learning_rate(opt.iterations)
                else:
                    lr = opt.learning_rate
                lr_value = float(tf.keras.backend.get_value(lr))
                logs[f'lr_optimizer_{i}'] = float(lr_value)

                with self.writer.as_default():
                    tf.summary.scalar(f'learning_rate_optimizer_{i}', data=lr_value, step=epoch)
                 #   self.writer.flush()
                    
        else:
            # Handle single optimizer case
            try:
                if isinstance(optimizer.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
                    lr = optimizer.learning_rate(optimizer.iterations)
                else:
                    lr = optimizer.learning_rate
                lr_value = float(tf.keras.backend.get_value(lr))
                logs[f'lr'] = float(lr_value)

                with self.writer.as_default():
                    tf.summary.scalar('learning_rate', data=lr_value, step=epoch)
                   # self.writer.flush()
            except Exception as e:
                print(f"Error logging learning rate: {e}")


class EarlyStoppingLogger(tf.keras.callbacks.EarlyStopping):
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        val_weighted_f1_score = logs.get('val_weighted_f1_score', None)
        if val_weighted_f1_score is not None:
            print(f" Epoch {epoch + 1}: val_weighted_f1_score = {val_weighted_f1_score:.4f}, patience = {self.wait}")
        else:
            print(f" Epoch {epoch + 1}: val_weighted_f1_score not available.")

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.precision = tf.keras.metrics.Precision(thresholds=threshold)
        self.recall = tf.keras.metrics.Recall(thresholds=threshold)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

    def get_config(self):
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config

class F1ScoreMultiClass(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', average='macro', threshold=0.5, num_classes=None, **kwargs):
        """
        Multi-class F1 score metric with support for various averaging methods.

        Args:
            name (str): Name of the metric.
            average (str): Averaging method - "macro", "weighted", or "micro".
            threshold (float): Threshold for binary classification (for each class in multi-class case).
            num_classes (int): Number of classes (required for "macro" or "weighted").
        """
        super().__init__(name=name, **kwargs)
        self.average = average
        self.threshold = threshold
        self.num_classes = num_classes

        # Initialize precision and recall per class
        self.precision = [tf.keras.metrics.Precision(thresholds=threshold) for _ in range(num_classes)]
        self.recall = [tf.keras.metrics.Recall(thresholds=threshold) for _ in range(num_classes)]

        # Accumulators for "micro" averaging
        if average == "micro":
            self.micro_tp = self.add_weight(name="micro_tp", initializer="zeros")
            self.micro_fp = self.add_weight(name="micro_fp", initializer="zeros")
            self.micro_fn = self.add_weight(name="micro_fn", initializer="zeros")

        # Weight storage for "weighted" averaging
        if average == "weighted":
            self.class_weights = self.add_weight(name="class_weights", shape=(num_classes,), initializer="zeros")
            self.class_totals = self.add_weight(name="class_totals", shape=(num_classes,), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update metric state with true and predicted labels.
        """
        # Ensure y_true and y_pred are the correct shape and type
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.float32)

        # If predictions are probabilities, apply thresholding
        if y_pred.shape[-1] == self.num_classes:  # Probabilistic predictions
            y_pred = tf.argmax(y_pred, axis=-1)

        y_true = tf.one_hot(y_true, depth=self.num_classes)
        y_pred = tf.one_hot(y_pred, depth=self.num_classes)

        if self.average == "micro":
            # Micro-averaged counts
            tp = tf.reduce_sum(y_true * y_pred)
            fp = tf.reduce_sum((1 - y_true) * y_pred)
            fn = tf.reduce_sum(y_true * (1 - y_pred))
            self.micro_tp.assign_add(tp)
            self.micro_fp.assign_add(fp)
            self.micro_fn.assign_add(fn)
        else:
            # Update per-class precision and recall
            for i in range(self.num_classes):
                self.precision[i].update_state(y_true[..., i], y_pred[..., i], sample_weight)
                self.recall[i].update_state(y_true[..., i], y_pred[..., i], sample_weight)

                if self.average == "weighted":
                    # Track weights for weighted averaging
                    class_total = tf.reduce_sum(y_true[..., i])
                    self.class_weights[i].assign_add(class_total)
                    self.class_totals[i].assign_add(1.0)

    def result(self):
        """
        Compute the F1 score based on the averaging method.
        """
        if self.average == "micro":
            # Micro-averaged F1
            precision = self.micro_tp / (self.micro_tp + self.micro_fp + tf.keras.backend.epsilon())
            recall = self.micro_tp / (self.micro_tp + self.micro_fn + tf.keras.backend.epsilon())
            return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        else:
            # Compute F1 for each class
            f1_scores = []
            for i in range(self.num_classes):
                p = self.precision[i].result()
                r = self.recall[i].result()
                f1 = 2 * (p * r) / (p + r + tf.keras.backend.epsilon())
                f1_scores.append(f1)

            f1_scores = tf.stack(f1_scores)

            if self.average == "macro":
                return tf.reduce_mean(f1_scores)
            elif self.average == "weighted":
                weights = self.class_weights / (tf.reduce_sum(self.class_weights) + tf.keras.backend.epsilon())
                return tf.reduce_sum(weights * f1_scores)

    def reset_state(self):
        """
        Reset all metrics and accumulators.
        """
        for i in range(self.num_classes):
            self.precision[i].reset_state()
            self.recall[i].reset_state()
        if self.average == "micro":
            self.micro_tp.assign(0.0)
            self.micro_fp.assign(0.0)
            self.micro_fn.assign(0.0)
        if self.average == "weighted":
            self.class_weights.assign(tf.zeros_like(self.class_weights))
            self.class_totals.assign(tf.zeros_like(self.class_totals))

    def get_config(self):
        """
        Return the configuration of the metric.
        """
        config = super().get_config()
        config.update({
            "average": self.average,
            "threshold": self.threshold,
            "num_classes": self.num_classes
        })
        return config


class CombinedAccuracyF1Metrics(tf.keras.callbacks.Callback):
    def __init__(self, x_val, y_val, y_val_stroke, val_sample_weights=None, 
                 thresholds=[0.5, 0.6, 0.7, 0.75, 0.8], name_prefix='val'):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val 
        self.y_val_stroke = y_val_stroke
        self.val_sample_weights = val_sample_weights
        self.thresholds = thresholds
        self.name_prefix = name_prefix

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.x_val, verbose=0)
        
        # Extract predictions and ground truths
        style_pred = np.argmax(y_pred[0], axis=-1)
        style_true = np.argmax(self.y_val, axis=-1)
        stroke_true = self.y_val_stroke.astype(int)
        
        # Style classification metrics
        style_pred_flat = style_pred.flatten()
        style_true_flat = style_true.flatten()
        style_acc = np.mean(style_pred_flat == style_true_flat)
        
        # Real stroke count
        stroke_true = (stroke_true >= 0.95).astype(int)  # Convert to binary
        stroke_true_flat = stroke_true.flatten()
        real_stroke_count = np.sum(stroke_true_flat, dtype=np.int32)
        
        # Prepare sample weights if available
        stroke_true_flat = stroke_true.flatten()
        sample_weights_flat = None
        if self.val_sample_weights is not None:
            sample_weights_flat = self.val_sample_weights.flatten()
        
        print(f"\n=== EPOCH {epoch + 1} VALIDATION METRICS ===")
        print(f"Style Accuracy: {style_acc:.4f}")
        print(f"Real Stroke Count: {real_stroke_count}")
        
        # Test multiple thresholds
        best_f1 = 0
        best_threshold = 0.5
        
        for thresh in self.thresholds:
            stroke_pred = (y_pred[1] > thresh).astype(int)
            stroke_pred_flat = stroke_pred.flatten()
            fake_stroke_count = np.sum(stroke_pred)
            
            # Unweighted metrics
            stroke_precision = precision_score(stroke_true_flat, stroke_pred_flat, 
                                             average='binary', zero_division=0)
            stroke_recall = recall_score(stroke_true_flat, stroke_pred_flat, 
                                       average='binary', zero_division=0)
            stroke_f1 = f1_score(stroke_true_flat, stroke_pred_flat, 
                                average='binary', zero_division=0)
            stroke_acc = np.mean(stroke_pred_flat == stroke_true_flat)
            
            # Weighted metrics (if sample weights available)
            if sample_weights_flat is not None:
                stroke_precision_weighted = precision_score(stroke_true_flat, stroke_pred_flat, 
                                                          sample_weight=sample_weights_flat,
                                                          average='binary', zero_division=0)
                stroke_recall_weighted = recall_score(stroke_true_flat, stroke_pred_flat, 
                                                     sample_weight=sample_weights_flat,
                                                     average='binary', zero_division=0)
                stroke_f1_weighted = f1_score(stroke_true_flat, stroke_pred_flat, 
                                             sample_weight=sample_weights_flat,
                                             average='binary', zero_division=0)
            else:
                stroke_precision_weighted = stroke_precision
                stroke_recall_weighted = stroke_recall
                stroke_f1_weighted = stroke_f1
            
            print(f"\nThreshold {thresh:.2f}:")
            print(f"  Predicted Strokes: {fake_stroke_count}")
            print(f"  Accuracy: {stroke_acc:.4f}")
            print(f"  Unweighted   - F1: {stroke_f1:.4f}, Precision: {stroke_precision:.4f}, Recall: {stroke_recall:.4f}")
            print(f"  Weighted     - F1: {stroke_f1_weighted:.4f}, Precision: {stroke_precision_weighted:.4f}, Recall: {stroke_recall_weighted:.4f}")
            
            # Track best threshold based on weighted F1
            if stroke_f1_weighted > best_f1:
                best_f1 = stroke_f1_weighted
                best_threshold = thresh
        
        # Log the best threshold metrics to logs
        stroke_pred_best = (y_pred[1] > best_threshold).astype(int)
        stroke_pred_flat_best = stroke_pred_best.flatten()
        
        stroke_precision_best = precision_score(stroke_true_flat, stroke_pred_flat_best, 
                                               average='binary', zero_division=0)
        stroke_recall_best = recall_score(stroke_true_flat, stroke_pred_flat_best, 
                                         average='binary', zero_division=0)
        stroke_f1_best = f1_score(stroke_true_flat, stroke_pred_flat_best, 
                                 average='binary', zero_division=0)
        
        if sample_weights_flat is not None:
            stroke_precision_weighted_best = precision_score(stroke_true_flat, stroke_pred_flat_best, 
                                                            sample_weight=sample_weights_flat,
                                                            average='binary', zero_division=0)
            stroke_recall_weighted_best = recall_score(stroke_true_flat, stroke_pred_flat_best, 
                                                      sample_weight=sample_weights_flat,
                                                      average='binary', zero_division=0)
            stroke_f1_weighted_best = f1_score(stroke_true_flat, stroke_pred_flat_best, 
                                              sample_weight=sample_weights_flat,
                                              average='binary', zero_division=0)
        else:
            stroke_precision_weighted_best = stroke_precision_best
            stroke_recall_weighted_best = stroke_recall_best
            stroke_f1_weighted_best = stroke_f1_best
        
        logs = logs or {}
        logs[f'{self.name_prefix}_style_acc'] = style_acc
        logs[f'{self.name_prefix}_best_threshold'] = best_threshold
        logs[f'{self.name_prefix}_stroke_f1_best'] = stroke_f1_best
        logs[f'{self.name_prefix}_stroke_precision_best'] = stroke_precision_best
        logs[f'{self.name_prefix}_stroke_recall_best'] = stroke_recall_best
        logs[f'{self.name_prefix}_stroke_f1_weighted_best'] = stroke_f1_weighted_best
        logs[f'{self.name_prefix}_stroke_precision_weighted_best'] = stroke_precision_weighted_best
        logs[f'{self.name_prefix}_stroke_recall_weighted_best'] = stroke_recall_weighted_best
        
        print(f"\n*** BEST THRESHOLD: {best_threshold:.2f} (Weighted F1: {best_f1:.4f}) ***")
        print("=" * 50)

def combined_metric(logs, alpha=0.5):
    """
    Combine two metrics with a weighted harmonic mean.
    
    :param logs: Dictionary containing logged metrics (e.g., logs from callbacks).
    :param alpha: Weight for the first metric (0 ≤ alpha ≤ 1). The second metric weight will be 1 - alpha.
    :return: Combined metric value.
    """
    metric1 = logs.get('val_stroke_label_output_weighted_f1_score', 0.0)  # Stroke branch
    metric2 = logs.get('val_swim_style_output_weighted_categorical_accuracy', 0.0)  # Swim style branch
    
    # Avoid division by zero
    if metric1 == 0 or metric2 == 0:
        return 0.0
    
    # Calculate the weighted harmonic mean
    harmonic_mean = 2 / ((alpha / metric1) + ((1 - alpha) / metric2))
    return harmonic_mean

class CombinedMetricCallback(tf.keras.callbacks.Callback):
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        combined = combined_metric(logs, alpha=self.alpha)
        logs['val_combined_metric'] = combined
        print(f"Epoch {epoch + 1}: val_combined_metric = {combined:.4f}")

class CombinedEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, monitor1, monitor2, mode1='max', mode2='max', patience=5, restore_best_weights=True):
        super(CombinedEarlyStopping, self).__init__()
        self.monitor1 = monitor1
        self.monitor2 = monitor2
        self.mode1 = mode1
        self.mode2 = mode2
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.best1 = -float('inf') if mode1 == 'max' else float('inf')
        self.best2 = -float('inf') if mode2 == 'max' else float('inf')
        self.wait = 0
        self.best_weights = None
        self.stopped_epoch = None  # Initialize this attribute

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current1 = logs.get(self.monitor1)
        current2 = logs.get(self.monitor2)

        if current1 is None or current2 is None:
            print(f"Warning: Monitor metrics {self.monitor1} or {self.monitor2} not found in logs.")
            return

        improved1 = (current1 > self.best1 if self.mode1 == 'max' else current1 < self.best1)
        improved2 = (current2 > self.best2 if self.mode2 == 'max' else current2 < self.best2)

        if improved1 or improved2:
            self.best1 = max(current1, self.best1) if self.mode1 == 'max' else min(current1, self.best1)
            self.best2 = max(current2, self.best2) if self.mode2 == 'max' else min(current2, self.best2)
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.restore_best_weights and self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if hasattr(self, 'stopped_epoch') and self.stopped_epoch is not None:
            print(f"Early stopping at epoch {self.stopped_epoch + 1}.")

class DebugCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch + 1} Logs:")
        for key, value in logs.items():
            print(f"{key}: {value}")

def get_callbacks(model_save_path, user, log_dir):
    callbacks = [
        # Early Stopping Logger
        EarlyStoppingLogger(
            monitor='val_weighted_f1_score',  # Use weighted F1 score if appropriate
            patience=30,
            restore_best_weights=True,
            min_delta=0.001,
            verbose=1,
            mode='max'
        ),
        # Model Checkpoint - More specific
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_save_path, f'model_{user}_best'),
            monitor='val_weighted_f1_score',  # Specific to stroke output
            save_best_only=True,
            mode='max',  # Changed to max
            verbose=1,
            save_weights_only=False  # Save full model
        ),
   
        # Reduce Learning Rate on Plateau - More conservative
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_weighted_f1_score',  # Align with early stopping
            factor=0.4,  # Slightly less aggressive reduction
            patience=10,  # Reduced patience
            min_delta=0.001,
            verbose=1,
            mode='max',  # Changed to max
            min_lr=5e-5  # Slightly higher minimum learning rate
        ),
        # TensorBoard with more detailed configuration
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch',
            profile_batch=0  # Disable profiling
        ),
        
        # Updated Learning Rate Logger
        LearningRateLogger_new(log_dir=log_dir),
        
        # CSV Logger with append mode
        tf.keras.callbacks.CSVLogger(
            os.path.join(model_save_path, f'training_history_{user}.csv'),
            separator=',',
            append=True  # Preserve history
        ),

        # Debug Callback
        DebugCallback()
    ]
    
    return callbacks

def get_callbacks_combined(model_save_path, user, log_dir):
    callbacks = [
    #    CombinedMetricCallback(alpha=0.7),
     #   CombinedEarlyStopping(
      #      monitor1='val_stroke_label_output_weighted_f1_score',
       #     monitor2='val_swim_style_output_weighted_categorical_accuracy',
        #    mode1='max',
         #   mode2='max',
      #      patience=30,
       #     restore_best_weights=True
     #   ),

        # Or with detailed logging
     #   TaskSpecificLRReducer([
      #      (0, 'val_loss', 'min', 0.25, 10, 1e-6),
       #     (1, 'val_swim_style_output_weighted_categorical_accuracy', 'max', 0.25, 10, 1e-6),
        #    (2, 'val_stroke_label_output_weighted_f1_score', 'max', 0.25, 10, 1e-6)
      #  ], verbose=2),
    
     #   MultiOptimizerLRReducer(
      #      monitor='val_stroke_label_output_weighted_f1_score',
      #      factor=0.4,
      #      patience=10,
      #      verbose=1,
      #      mode='max',  # Changed to max
      #      min_lr=1e-6,
      #      min_delta=0.001
      #  ),
        # Model Checkpoint - More specific
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_save_path, f'model_{user}_best'),
            monitor='val_combined_metric',  # Specific to stroke output
            save_best_only=True,
            mode='max',  # Changed to max
            verbose=1,
            save_weights_only=False  # Save full model
        ),
        
        # TensorBoard with more detailed configuration
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch',
            profile_batch=0  # Disable profiling
        ),
        
        # Updated Learning Rate Logger
        LearningRateLogger_new(log_dir=log_dir),
        
        # CSV Logger with append mode
        tf.keras.callbacks.CSVLogger(
            os.path.join(model_save_path, f'training_history_{user}.csv'),
            separator=',',
            append=True  # Preserve history
        ),
        # Debug Callback
       # DebugCallback()
    ]
    
    return callbacks

def get_callbacks_regression(model_save_path, user, log_dir, x_val, y_val, y_stroke_val):
    callbacks = [
        CombinedMetricCallback(alpha=0.5),

        # Or with detailed logging
        TaskSpecificLRReducer([
            (0, 'val_loss', 'min', 0.5, 10, 1e-6),
            (1, 'val_swim_style_output_weighted_categorical_accuracy', 'max', 0.5, 10, 1e-6),
            (2, 'val_stroke_label_output_weighted_f1_score', 'max', 0.5, 10, 1e-6)
        ], verbose=2),
        CombinedAccuracyF1Metrics(x_val, y_val, y_stroke_val),
     #   MultiOptimizerLRReducer(
      #      monitor='val_stroke_label_output_weighted_f1_score',
      #      factor=0.4,
      #      patience=10,
      #      verbose=1,
      #      mode='max',  # Changed to max
      #      min_lr=1e-6,
      #      min_delta=0.001
      #  ),
        # Model Checkpoint - More specific
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_save_path, f'model_{user}_best'),
            monitor='val_combined_metric',  # Specific to stroke output
            save_best_only=True,
            mode='max',  # Changed to max
            verbose=1,
            save_weights_only=False  # Save full model
        ),
        
        # TensorBoard with more detailed configuration
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch',
            profile_batch=0  # Disable profiling
        ),
        
        # Updated Learning Rate Logger
        LearningRateLogger_new(log_dir=log_dir),
        
        # CSV Logger with append mode
        tf.keras.callbacks.CSVLogger(
            os.path.join(model_save_path, f'training_history_{user}.csv'),
            separator=',',
            append=True  # Preserve history
        ),
        # Debug Callback
        DebugCallback()
    ]
    
    return callbacks

def get_layers(model, layer_name_filter=None):
    """
    Recursively retrieve layers from a model, including wrapped layers in constructs like TimeDistributed or Bidirectional.
    
    :param model: The model or layer to retrieve layers from.
    :param layer_name_filter: A substring to filter layers by name.
    :return: A list of layers matching the filter.
    """
    layers = []
    if hasattr(model, 'layers'):  # Check if the model or layer has sub-layers
        for layer in model.layers:
            layers.extend(get_layers(layer, layer_name_filter))
    else:
        layers = [model]  # Base case: This is a single layer

    # Apply name filter if provided
    if layer_name_filter:
        layers = [layer for layer in layers if layer_name_filter in layer.name]
    return layers


class MultiOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, optimizers_and_layers):
        super(MultiOptimizer, self).__init__(name="MultiOptimizer")
        self.optimizers_and_layers = optimizers_and_layers
        
        # Set a default learning rate for the main optimizer
        self._learning_rate = self.optimizers_and_layers[0][0].learning_rate
        
    def _create_slots(self, var_list):
        for optimizer, _ in self.optimizers_and_layers:
            optimizer._create_slots(var_list)
            
    def apply_gradients(self, grads_and_vars, **kwargs):
        optimizer_grads = {opt: [] for opt, _ in self.optimizers_and_layers}
        
        for grad, var in grads_and_vars:
            for optimizer, layers in self.optimizers_and_layers:
                if any(var.name.startswith(layer.name) for layer in layers):
                    optimizer_grads[optimizer].append((grad, var))
                    break
                    
        for optimizer, grads in optimizer_grads.items():
            if grads:
                optimizer.apply_gradients(grads, **kwargs)

    @property
    def learning_rate(self):
        # Return the swim style learning rate for logging
        return self._learning_rate

    def get_config(self):
        return {"name": self.name}


def compile_model(data_parameters, model, training_parameters, loss_parameters=None):

    if data_parameters['label_type'] == 'sparse':
        type_categorical_crossentropy = tf.keras.losses.SparseCategoricalCrossentropy()#'sparse_categorical_crossentropy'
    else:
        type_categorical_crossentropy = tf.keras.losses.CategoricalCrossentropy()
                                        #tf.keras.losses.CategoricalFocalCrossentropy(
                                         #                   gamma=2.0,
                                          #                  from_logits=False,
                                           #                 label_smoothing=0.0) #tf.keras.losses.CategoricalCrossentropy() #'categorical_crossentropy'

    if training_parameters['swim_style_output'] and training_parameters['stroke_label_output']:
        # Get layers for each branch
        swim_style_layers = [layer for layer in model.layers 
                            if 'swim_style' in layer.name]
        stroke_layers     = [layer for layer in model.layers 
                            if 'stroke' in layer.name]
        common_layers = [layer for layer in model.layers
                        if 'swim_style' not in layer.name and 'stroke' not in layer.name
                        and not isinstance(layer, tf.keras.layers.InputLayer)]

         # Debug print
        if data_parameters['debug']:
            print("\nSwim Style Layers:")
            for layer in swim_style_layers:
                print(f"  {layer.name, layer.trainable}")
        
            print("\nStroke Layers:")
            for layer in stroke_layers:
                print(f"  {layer.name, layer.trainable}")

            print("\nCommon Layers:")
            for layer in common_layers:
                print(f"  {layer.name, layer.trainable}")

        if 'common_lr' in training_parameters:
            common_optimizer = tf.keras.optimizers.Adam(
                learning_rate=training_parameters['common_lr'],  # Slightly lower than task-specific rates
                beta_1=training_parameters['beta_1'],
                beta_2=training_parameters['beta_2']
            )
        # Create optimizers with explicit learning rates
        swim_style_optimizer = tf.keras.optimizers.Adam(
            learning_rate=training_parameters['swim_style_lr'],  
            beta_1=training_parameters['beta_1'],
            beta_2=training_parameters['beta_2']
        )

        stroke_optimizer = tf.keras.optimizers.Adam(
            learning_rate=training_parameters['stroke_lr'],  
            beta_1=training_parameters['beta_1'],
            beta_2=training_parameters['beta_2']
        )
        """
        stroke_optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=training_parameters['stroke_lr']['initial_lr'],
                decay_steps=training_parameters['stroke_lr']['decay_steps'],
                decay_rate=training_parameters['stroke_lr']['decay_rate']
            ),
            beta_1=training_parameters['beta_1'],
            beta_2=training_parameters['beta_2']
        )
        """
        # Combine optimizers
 #       optimizer = MultiOptimizer([
  #          (swim_style_optimizer, swim_style_layers),
   #         (stroke_optimizer, stroke_layers)
    #    ])
        #optimizers_and_layers = [(swim_style_optimizer, common_layers + swim_style_layers), (stroke_optimizer, common_layers + stroke_layers)]
        if 'common_lr' in training_parameters:
            optimizers_and_layers = [
                (common_optimizer, common_layers),
                (swim_style_optimizer, swim_style_layers),
                (stroke_optimizer, stroke_layers)
            ]
        else:
            optimizers_and_layers = [(swim_style_optimizer, common_layers + swim_style_layers), (stroke_optimizer, common_layers + stroke_layers)]
        
        optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
        loss={
            'swim_style_output': type_categorical_crossentropy,
            'stroke_label_output': ConsistentPeakBFCE(
                alpha=loss_parameters['alpha'],
                gamma=loss_parameters['gamma'],
                target_height=loss_parameters['target_height'],
                peak_weight=loss_parameters['peak_weight'],
                peak_threshold=loss_parameters['peak_threshold']
            )#tf.keras.losses.BinaryCrossentropy()
                                #weighted_binary_crossentropy_smooth_class(class_weights)
                              #  focal_loss(gamma=2.0, alpha=0.5)
                              #  tf.keras.losses.BinaryFocalCrossentropy(    
                               #             apply_class_balancing=False,
                                #            alpha=0.85,
                                 #           gamma=1.25,
                                  #          from_logits=False,
                                   #         label_smoothing=0.00)
        }
        sample_weight_mode={
            'swim_style_output': None,
            'stroke_label_output': 'temporal'  # For timestep-level weights
        }
        metrics={
            'swim_style_output': [
                tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ],
            'stroke_label_output': [
                tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5),
             #   tf.keras.metrics.MeanAbsoluteError(name='mae'),
              #  tf.keras.metrics.MeanSquaredError(name='mse'),
                tf.keras.metrics.Precision(name='precision', thresholds=0.5),
                tf.keras.metrics.Recall(name='recall', thresholds=0.5),
                F1Score(name='f1_score', threshold=0.5)
            ]
        }
        weighted_metrics={
            'swim_style_output': [
                tf.keras.metrics.CategoricalAccuracy(name='weighted_categorical_accuracy'),
                tf.keras.metrics.Precision(name='weighted_precision'),
                tf.keras.metrics.Recall(name='weighted_recall')
            ],
            'stroke_label_output': [
                tf.keras.metrics.BinaryAccuracy(name='weighted_accuracy', threshold=0.5),
            #    tf.keras.metrics.MeanAbsoluteError(name='weighted_mae'),
            #    tf.keras.metrics.MeanSquaredError(name='weighted_mse'),
                tf.keras.metrics.Precision(name='weighted_precision', thresholds=0.5),
                tf.keras.metrics.Recall(name='weighted_recall', thresholds=0.5),
                F1Score(name='weighted_f1_score', threshold=0.5)
            ]
        }

    elif training_parameters['swim_style_output']:
        # Create optimizers with explicit learning rates
        swim_style_optimizer = tf.keras.optimizers.Adam(
            learning_rate=training_parameters['swim_style_lr'],  
            beta_1=training_parameters['beta_1'],
            beta_2=training_parameters['beta_2']
        )
        optimizer = swim_style_optimizer
        loss = {
            'swim_style_output': type_categorical_crossentropy
        }
        sample_weight_mode={
            'swim_style_output': None
        }
        metrics = {
            'swim_style_output': [
                tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
                tf.keras.metrics.CategoricalCrossentropy(name='cross_entropy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        }
        weighted_metrics = {
            'swim_style_output': [
                tf.keras.metrics.CategoricalAccuracy(name='weighted_categorical_accuracy'),
                tf.keras.metrics.Precision(name='weighted_precision'),
                tf.keras.metrics.Recall(name='weighted_recall')
            ]
        }
    else:
        stroke_optimizer = tf.keras.optimizers.Adam(
            learning_rate=training_parameters['stroke_lr'],  
            beta_1=training_parameters['beta_1'],
            beta_2=training_parameters['beta_2']
        )
        """
        stroke_optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=training_parameters['stroke_lr']['initial_lr'],
                decay_steps=training_parameters['stroke_lr']['decay_steps'],
                decay_rate=training_parameters['stroke_lr']['decay_rate']
            ),
            beta_1=training_parameters['beta_1'],
            beta_2=training_parameters['beta_2']
        )
        """
        optimizer = stroke_optimizer
        loss = {
            'stroke_label_output': tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1)
#weighted_binary_crossentropy_smooth_class(class_weights)
        }
        sample_weight_mode={
            'stroke_label_output': 'temporal'  # For timestep-level weights
        }
        metrics = {
            'stroke_label_output': [
                tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5),
                tf.keras.metrics.Precision(name='precision', thresholds=0.5),
                tf.keras.metrics.Recall(name='recall', thresholds=0.5),
                F1Score(name='f1_score', threshold=0.5)
            ]
        }
        weighted_metrics = {
            'stroke_label_output': [
                tf.keras.metrics.BinaryAccuracy(name='weighted_accuracy', threshold=0.5),
                tf.keras.metrics.Precision(name='weighted_precision', thresholds=0.5),
                tf.keras.metrics.Recall(name='weighted_recall', thresholds=0.5),
                F1Score(name='weighted_f1_score', threshold=0.5)
            ]
        }

    # Then use it in model compilation
    model.compile(
        optimizer=optimizer,
        loss=loss,
        loss_weights={'swim_style_output': 1.0, 'stroke_label_output': 1.0},
   #     sample_weight_mode=sample_weight_mode,
        metrics=metrics,
        weighted_metrics=weighted_metrics

    )

    return model

def compile_regression_model(model, training_parameters):
    huber_loss = tf.keras.losses.Huber()
    common_optimizer = tf.keras.optimizers.Adam(
        learning_rate=training_parameters['common_lr'],  
        beta_1=training_parameters['beta_1'],
        beta_2=training_parameters['beta_2']
    )
    metrics = [
        tf.keras.metrics.MeanAbsoluteError(name='mae'),
        tf.keras.metrics.MeanSquaredError(name='mse'),
    ]
    model.compile(
        optimizer = common_optimizer,
        loss = huber_loss,
        metrics = metrics
    )
    return model
