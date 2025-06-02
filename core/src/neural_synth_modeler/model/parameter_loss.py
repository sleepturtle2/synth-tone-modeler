import tensorflow as tf

class ParameterLoss(tf.keras.losses.Loss):
    def __init__(self, osc_weight=1.0, filter_weight=1.0, fx_weight=1.0):
        super().__init__()
        self.osc_weight = osc_weight
        self.filter_weight = filter_weight
        self.fx_weight = fx_weight

    def call(self, y_true, y_pred):
        # Handle y_true
        if isinstance(y_true, dict):
            osc_true = y_true['oscillators']
            filter_true = y_true['filters']
            fx_true = y_true['fx']
        else:
            # Get the number of parameters for each component
            num_osc = tf.shape(y_pred['oscillators'])[-1] if isinstance(y_pred, dict) else y_pred.shape[-1] // 3
            num_filter = tf.shape(y_pred['filters'])[-1] if isinstance(y_pred, dict) else y_pred.shape[-1] // 3
            osc_true, filter_true, fx_true = tf.split(y_true, [num_osc, num_filter, -1], axis=-1)

        # Handle y_pred
        if isinstance(y_pred, dict):
            osc_pred = y_pred['oscillators']
            filter_pred = y_pred['filters']
            fx_pred = y_pred['fx']
        else:
            num_osc = tf.shape(osc_true)[-1]
            num_filter = tf.shape(filter_true)[-1]
            osc_pred, filter_pred, fx_pred = tf.split(y_pred, [num_osc, num_filter, -1], axis=-1)

        # Calculate losses
        osc_loss = tf.reduce_mean(tf.square(osc_true - osc_pred))
        filter_loss = tf.reduce_mean(tf.square(filter_true - filter_pred))
        fx_loss = tf.reduce_mean(tf.square(fx_true - fx_pred))
        
        return (self.osc_weight * osc_loss + 
                self.filter_weight * filter_loss + 
                self.fx_weight * fx_loss)

    def get_config(self):
        return {
            'osc_weight': self.osc_weight,
            'filter_weight': self.filter_weight,
            'fx_weight': self.fx_weight
        }