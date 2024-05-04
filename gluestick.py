import tensorflow as tf
from tensorflow.keras import layers, models


def MLP(channels, do_bn=True):
    mlp = models.Sequential()
    mlp.add(layers.InputLayer(shape=(None, channels[0])))
    for num_features in channels[:-1]:
        mlp.add(layers.Conv1D(num_features, kernel_size=1, use_bias=True))
        if do_bn:
            mlp.add(layers.BatchNormalization())
        mlp.add(layers.ReLU())
    mlp.add(layers.Conv1D(channels[-1], kernel_size=1, use_bias=True))
    return mlp


class KeypointEncoder(models.Model):
    def __init__(self, desc_dim, kp_encoder):
        super().__init__()
        self.encoder = MLP([3] + list(kp_encoder) + [desc_dim], do_bn=True)

    def call(self, kpts, scores):
        inputs = tf.concat((kpts, scores[..., None]), axis=-1)
        return self.encoder(inputs)


class EndPtEncoder(models.Model):
    def __init__(self, desc_dim, kp_encoder):
        super().__init__()
        self.encoder = MLP([5] + list(kp_encoder) + [desc_dim], do_bn=True)

    def call(self, endpoints, scores):
        b_size, n_pts, _, _ = endpoints.shape
        endpt_offset = (endpoints[:, :, 1] - endpoints[:, :, 0])[:, :, None]
        endpt_offset = tf.concat((endpt_offset, -endpt_offset), axis=2)
        endpt_offset = tf.reshape(endpt_offset, (b_size, 2 * n_pts, 2))
        inputs = [tf.reshape(endpoints, (b_size, -1, 2)), endpt_offset, tf.tile(scores, [1, 2])[..., None]]
        return self.encoder(tf.concat(inputs, axis=-1))


def attention(query, key, value):
    # note - dimensions channel first
    dim = tf.cast(tf.shape(query)[1], dtype=tf.float32)
    scores = tf.einsum("bdhn,bdhm->bhnm", query, key) / tf.math.sqrt(dim)
    prob = tf.nn.softmax(scores, axis=-1)
    output = tf.einsum("bhnm,bdhm->bdhn", prob, value)
    return output, prob


class MultiHeadedAttention(models.Model):
    def __init__(self, h, d_model):
        super().__init__()
        assert d_model % h == 0
        self.dim = d_model // h
        self.h = h
        self.merge = layers.Conv1D(d_model, kernel_size=1, input_shape=(None, d_model))
        self.proj_q = layers.Conv1D(d_model, kernel_size=1, input_shape=(None, d_model))
        self.proj_k = layers.Conv1D(d_model, kernel_size=1, input_shape=(None, d_model))
        self.proj_v = layers.Conv1D(d_model, kernel_size=1, input_shape=(None, d_model))

    def call(self, query, key, value):
        b = query.shape[0]
        proj_query = tf.reshape(tf.transpose(self.proj_q(query), (0, 2, 1)), (b, self.dim, self.h, -1))
        proj_key = tf.reshape(tf.transpose(self.proj_k(key), (0, 2, 1)), (b, self.dim, self.h, -1))
        proj_value = tf.reshape(tf.transpose(self.proj_v(value), (0, 2, 1)), (b, self.dim, self.h, -1))
        x, prob = attention(proj_query, proj_key, proj_value)
        x = tf.reshape(x, (b, self.dim * self.h, -1))
        # back to channel last
        x = tf.transpose(x, (0, 2, 1))
        return self.merge(x)


class AttentionalPropagation(models.Model):
    def __init__(self, num_dim, num_heads, skip_init=False):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, num_dim)
        self.mlp = MLP([num_dim * 2, num_dim * 2, num_dim], do_bn=True)
        if skip_init:
            pass
            # self.register_parameter("scaling", nn.Parameter(torch.tensor(0.0)))
        else:
            self.scaling = 1.0

    def call(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(tf.concat([x, message], axis=-1)) * self.scaling


class GNNLayer(models.Model):
    def __init__(self, feature_dim, layer_type, skip_init):
        super().__init__()
        assert layer_type in ["cross", "self"]
        self.type = layer_type
        self.update = AttentionalPropagation(feature_dim, 4, skip_init)

    def call(self, desc0, desc1):
        if self.type == "cross":
            src0, src1 = desc1, desc0
        elif self.type == "self":
            src0, src1 = desc0, desc1
        else:
            raise ValueError("Unknown layer type: " + self.type)
        # self.update.attn.prob = []
        delta0, delta1 = self.update(desc0, src0), self.update(desc1, src1)
        desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


class LineLayer(models.Model):
    def __init__(self, feature_dim, line_attention=False):
        super().__init__()
        self.dim = feature_dim
        self.mlp = MLP([self.dim * 3, self.dim * 2, self.dim], do_bn=True)
        self.line_attention = line_attention
        if line_attention:
            self.proj_node = layers.Conv1d(self.dim, kernel_size=1, input_shape=(None, self.dim))
            self.proj_neigh = layers.Conv1d(self.dim, kernel_size=1, input_shape=(None, 2 * self.dim))

    def get_endpoint_update(self, ldesc, line_enc, lines_junc_idx):
        # ldesc is [bs, D, n_junc], line_enc [bs, D, n_lines * 2]
        # and lines_junc_idx [bs, n_lines * 2]
        # Create one message per line endpoint
        b_size, endpoints = lines_junc_idx.shape[0]

        batch_range = tf.range(b_size)[:, tf.newaxis]
        batch_range = tf.tile(batch_range, [1, endpoints])
        combined_indices = tf.stack([batch_range, lines_junc_idx], axis=-1)

        line_desc = tf.transpose(tf.gather_nd(tf.transpose(ldesc, (0, 2, 1)), combined_indices), (0, 2, 1))
        line_desc2 = tf.reverse(tf.reshape(line_desc, (b_size, self.dim, -1, 2)), axis=[-1])
        message = tf.concat([line_desc, tf.identity(tf.reshape(line_desc2, [b_size, self.dim, -1])), line_enc], axis=1)
        message = tf.transpose(message, (0, 2, 1))
        return self.mlp(message)  # [b_size, n_lines * 2, D]  # note it's channel last here

    def get_endpoint_attention(self, ldesc, line_enc, lines_junc_idx):
        # they're not using it
        pass

    def call(self, ldesc0, ldesc1, line_enc0, line_enc1, lines_junc_idx0, lines_junc_idx1):
        # Gather the endpoint updates
        lupdate0 = self.get_endpoint_update(ldesc0, line_enc0, lines_junc_idx0)
        lupdate1 = self.get_endpoint_update(ldesc1, line_enc1, lines_junc_idx1)

        if self.line_attention:
            raise NotImplementedError
        else:
            # Average the updates for each junction
            update0 = scatter_mean(ldesc0.shape, lines_junc_idx0, lupdate0)
            update1 = scatter_mean(ldesc1.shape, lines_junc_idx1, lupdate1)

            # Update
            ldesc0 = ldesc0 + update0
            ldesc1 = ldesc1 + update1

            return ldesc0, ldesc1


def scatter_mean(output_shape, indices, values):
    update0 = tf.zeros(output_shape)
    b_size, endpoints = indices.shape
    batch_range = tf.range(b_size)[:, tf.newaxis]
    batch_range = tf.tile(batch_range, [1, endpoints])
    combined_indices = tf.stack([batch_range, indices], axis=-1)
    scatter_counts = tf.tensor_scatter_nd_add(update0, combined_indices, tf.ones_like(values, dtype=tf.float32))
    scatter_sum = tf.tensor_scatter_nd_add(update0, combined_indices, values)
    return tf.divide(scatter_sum, tf.maximum(scatter_counts, 1))
