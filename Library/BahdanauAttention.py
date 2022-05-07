"""
Attention Block
"""

# Imports
import tensorflow as tf

# Main Functions
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        # For Eqn. (4), the    Bahdanau attention
        self.W1 = tf.keras.layers.Dense(units, use_bias=False)
        self.W2 = tf.keras.layers.Dense(units, use_bias=False)

        self.attention = tf.keras.layers.AdditiveAttention()

    def call(self, query, value, mask=None):
        # shape_checker = ShapeChecker()
        # shape_checker(query, ('batch', 't', 'query_units'))
        # shape_checker(value, ('batch', 's', 'value_units'))
        # shape_checker(mask, ('batch', 's'))

        # From Eqn. (4), `W1@ht`.
        w1_query = self.W1(query)
        # shape_checker(w1_query, ('batch', 't', 'attn_units'))

        # From Eqn. (4), `W2@hs`.
        w2_key = self.W2(value)
        # shape_checker(w2_key, ('batch', 's', 'attn_units'))

        query_mask = tf.ones(tf.shape(query)[:-1], dtype=bool)
        value_mask = mask if mask is not None else tf.ones(tf.shape(value)[:-1], dtype=bool)

        context_vector, attention_weights = self.attention(
                inputs = [w1_query, value, w2_key],
                mask=[query_mask, value_mask],
                return_attention_scores = True,
        )
        # shape_checker(context_vector, ('batch', 't', 'value_units'))
        # shape_checker(attention_weights, ('batch', 't', 's'))

        return context_vector, attention_weights

    def get_config(self):
        config = super(BahdanauAttention, self).get_config()
        config.update({"units": self.units})
        return config