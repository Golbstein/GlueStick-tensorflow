import dataclasses

import tensorflow as tf
from tensorflow.keras import layers, models


def dict_to_dataclass(config_dict):
    fields_list = []
    for key, value in config_dict.items():
        fields_list.append((key, type(value)))
    conf = dataclasses.make_dataclass('Config', fields_list)
    return conf(**config_dict)


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
        # ldesc is [bs, n_junc, D], line_enc [bs, n_lines * 2, D]
        # and lines_junc_idx [bs, n_lines * 2]
        # Create one message per line endpoint
        b_size = lines_junc_idx.shape[0]

        line_desc = tf.transpose(gather_like_in_torch(lines_junc_idx, ldesc), (0, 2, 1))  # channel first
        line_desc2 = tf.reverse(tf.reshape(line_desc, (b_size, self.dim, -1, 2)), axis=[-1])
        message = tf.concat([line_desc, tf.identity(tf.reshape(line_desc2, [b_size, self.dim, -1])), tf.transpose(line_enc, (0, 2, 1))], axis=1)
        message = tf.transpose(message, (0, 2, 1))  # back to channel last
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


def normalize_keypoints(kpts, h, w):
    size = tf.convert_to_tensor([[tf.cast(w, kpts.dtype), tf.cast(h, kpts.dtype)]])
    c = size / 2.
    f = tf.reduce_max(c, keepdims=True) * 0.7
    return (kpts - c[:, None, :]) / f[:, None, :]


class AttentionalGNN(models.Model):
    def __init__(self, feature_dim, layer_types, skip=False, num_line_iterations=1, line_attention=False):
        super().__init__()
        self.num_line_iterations = num_line_iterations
        self.inter_layers = {}
        self.gnn_layers = [GNNLayer(feature_dim, layer_type, skip) for layer_type in layer_types]
        self.line_layers = [LineLayer(feature_dim, line_attention) for _ in range(len(layer_types) // 2)]

    def call(self, desc0, desc1, line_enc0, line_enc1, lines_junc_idx0, lines_junc_idx1):
        for i, layer in enumerate(self.gnn_layers):
            desc0, desc1 = layer(desc0, desc1)
            if layer.type == "self" and lines_junc_idx0.shape[1] > 0 and lines_junc_idx1.shape[1] > 0:
                # Add line self attention layers after every self layer
                for _ in range(self.num_line_iterations):
                    desc0, desc1 = self.line_layers[i // 2](desc0, desc1, line_enc0, line_enc1, lines_junc_idx0, lines_junc_idx1)
        return desc0, desc1


class GlueStick(models.Model):
    default_conf = {
        "input_dim": 256,
        "descriptor_dim": 256,
        "weights": None,
        "version": "v0.1_arxiv",
        "keypoint_encoder": [32, 64, 128, 256],
        "GNN_layers": ["self", "cross"] * 9,
        "num_line_iterations": 1,
        "line_attention": False,
        "filter_threshold": 0.2,
        "skip_init": False,
        "inter_supervision": None,
        "loss": {
            "nll_weight": 1.0,
            "nll_balancing": 0.5,
            "inter_supervision": [0.3, 0.6],
        },
    }
    required_data_keys = [
        "view0",
        "view1",
        "keypoints0",
        "keypoints1",
        "descriptors0",
        "descriptors1",
        "keypoint_scores0",
        "keypoint_scores1",
        "lines0",
        "lines1",
        "lines_junc_idx0",
        "lines_junc_idx1",
        "line_scores0",
        "line_scores1",
    ]

    DEFAULT_LOSS_CONF = {"nll_weight": 1.0, "nll_balancing": 0.5}

    def __init__(self, conf=None):
        super().__init__(name='tf-GlueStick')
        if conf is None:
            self.conf = dict_to_dataclass(self.default_conf)
        else:
            self.conf = conf
        if self.conf.input_dim != self.conf.descriptor_dim:
            self.input_proj = layers.Conv1D(self.conf.descriptor_dim, kernel_size=1, input_shape=(None, self.conf.input_dim))
        self.kenc = KeypointEncoder(self.conf.descriptor_dim, self.conf.keypoint_encoder)
        self.lenc = EndPtEncoder(self.conf.descriptor_dim, self.conf.keypoint_encoder)
        self.gnn = AttentionalGNN(self.conf.descriptor_dim, self.conf.GNN_layers, num_line_iterations=self.conf.num_line_iterations, line_attention=self.conf.line_attention)
        self.final_proj = layers.Conv1D(self.conf.descriptor_dim, kernel_size=1, input_shape=(None, self.conf.descriptor_dim))
        self.final_line_proj = layers.Conv1D(self.conf.descriptor_dim, kernel_size=1, input_shape=(None, self.conf.descriptor_dim))
        self.bin_score = tf.Variable(1.0, trainable=True, name='bin_score')
        self.line_bin_score = tf.Variable(1.0, trainable=True, name='line_bin_score')

    def _get_matches(self, scores_mat):
        max0 = tf.reduce_max(scores_mat[:, :-1, :-1], axis=2)
        m0 = tf.argmax(scores_mat[:, :-1, :-1], axis=2, output_type=tf.int32)
        m1 = tf.argmax(scores_mat[:, :-1, :-1], axis=1, output_type=tf.int32)
        mutual0 = tf.range(m0.shape[1])[None] == gather_like_in_torch(m0, m1)
        mutual1 = tf.range(m1.shape[1])[None] == gather_like_in_torch(m1, m0)
        mscores0 = tf.where(mutual0, tf.exp(max0), 0)
        mscores1 = tf.where(mutual1, gather_like_in_torch(m1, mscores0), 0)
        valid0 = mutual0 & (mscores0 > self.conf.filter_threshold)
        valid1 = mutual1 & gather_like_in_torch(m1, valid0)
        m0 = tf.where(valid0, m0, -1)
        m1 = tf.where(valid1, m1, -1)
        return m0, m1, mscores0, mscores1

    def _get_line_matches(self, ldesc0, ldesc1, lines_junc_idx0, lines_junc_idx1, final_proj):
        mldesc0 = final_proj(ldesc0)
        mldesc1 = final_proj(ldesc1)
        line_scores = mldesc0 @ tf.transpose(mldesc1, (0, 2, 1))
        line_scores = line_scores / self.conf.descriptor_dim ** 0.5

        # Get the line representation from the junction descriptors
        n2_lines0 = lines_junc_idx0.shape[1]
        n2_lines1 = lines_junc_idx1.shape[1]

        line_scores = tf.transpose(gather_like_in_torch(lines_junc_idx1, tf.transpose(line_scores, (0, 2, 1))), (0, 2, 1))
        line_scores = gather_like_in_torch(lines_junc_idx0, line_scores)
        line_scores = tf.reshape(line_scores, (-1, n2_lines0 // 2, 2, n2_lines1 // 2, 2))

        # Match either in one direction or the other
        raw_line_scores = 0.5 * tf.maximum(line_scores[:, :, 0, :, 0] + line_scores[:, :, 1, :, 1], line_scores[:, :, 0, :, 1] + line_scores[:, :, 1, :, 0])

        line_scores = log_double_softmax(raw_line_scores, self.line_bin_score)
        m0_lines, m1_lines, mscores0_lines, mscores1_lines = self._get_matches(line_scores)
        return line_scores, m0_lines, m1_lines, mscores0_lines, mscores1_lines, raw_line_scores

    def call(self, data):
        b_size = len(data["keypoints0"])
        _, h0, w0, _ = data["view0"]["image"].shape
        _, h1, w1, _ = data["view1"]["image"].shape
        pred = {}
        desc0, desc1 = data["descriptors0"], data["descriptors1"]
        kpts0, kpts1 = data["keypoints0"], data["keypoints1"]
        n_kpts0, n_kpts1 = kpts0.shape[1], kpts1.shape[1]
        n_lines0, n_lines1 = data["lines0"].shape[1], data["lines1"].shape[1]

        lines0 = tf.reshape(data["lines0"], (-1, n_lines0 * 2, 2))
        lines1 = tf.reshape(data["lines1"], (-1, n_lines1 * 2, 2))

        lines_junc_idx0 = tf.cast(tf.reshape(data["lines_junc_idx0"], (-1, n_lines0 * 2)), dtype=tf.int32)
        lines_junc_idx1 = tf.cast(tf.reshape(data["lines_junc_idx1"], (-1, n_lines1 * 2)), dtype=tf.int32)

        if self.conf.input_dim != self.conf.descriptor_dim:
            desc0 = self.input_proj(desc0)
            desc1 = self.input_proj(desc1)

        kpts0 = normalize_keypoints(kpts0, h0, w0)
        kpts1 = normalize_keypoints(kpts1, h1, w1)

        desc0 = desc0 + self.kenc(kpts0, data["keypoint_scores0"])
        desc1 = desc1 + self.kenc(kpts1, data["keypoint_scores1"])

        if n_lines0 != 0 and n_lines1 != 0:
            lines0 = tf.reshape(normalize_keypoints(lines0, h0, w0), (b_size, n_lines0, 2, 2))
            lines1 = tf.reshape(normalize_keypoints(lines1, h1, w1), (b_size, n_lines1, 2, 2))
            line_enc0 = self.lenc(lines0, data["line_scores0"])
            line_enc1 = self.lenc(lines1, data["line_scores1"])
        else:
            line_enc0 = tf.zeros((b_size, self.conf.descriptor_dim, n_lines0 * 2), dtype=kpts0.dtype)
            line_enc1 = tf.zeros((b_size, self.conf.descriptor_dim, n_lines1 * 2), dtype=kpts1.dtype)

        desc0, desc1 = self.gnn(desc0, desc1, line_enc0, line_enc1, lines_junc_idx0, lines_junc_idx1)

        # Match all points (KP and line junctions)
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        kp_scores = mdesc0 @ tf.transpose(mdesc1, (0, 2, 1))
        kp_scores = kp_scores / self.conf.descriptor_dim ** 0.5
        m0, m1, mscores0, mscores1 = self._get_matches(kp_scores)
        pred["log_assignment"] = kp_scores
        pred["matches0"] = m0
        pred["matches1"] = m1
        pred["matching_scores0"] = mscores0
        pred["matching_scores1"] = mscores1

        # Match the lines
        if n_lines0 > 0 and n_lines1 > 0:
            line_scores, m0_lines, m1_lines, mscores0_lines, mscores1_lines, raw_line_scores = self._get_line_matches(
                desc0[:, :2 * n_lines0], desc1[:, :2 * n_lines1], lines_junc_idx0, lines_junc_idx1, self.final_line_proj)
        else:
            line_scores = tf.zeros((b_size, n_lines0, n_lines1), dtype=tf.float32)
            m0_lines = -tf.ones((b_size, n_lines0), dtype=tf.int64)
            m1_lines = -tf.ones((b_size, n_lines1), dtype=tf.int64)
            mscores0_lines = tf.zeros((b_size, n_lines0), dtype=tf.float32)
            mscores1_lines = tf.zeros((b_size, n_lines1), dtype=tf.float32)
            raw_line_scores = tf.zeros(b_size, n_lines0, n_lines1, dtype=tf.float32)

        pred["line_log_assignment"] = line_scores
        pred["line_matches0"] = m0_lines
        pred["line_matches1"] = m1_lines
        pred["line_matching_scores0"] = mscores0_lines
        pred["line_matching_scores1"] = mscores1_lines
        pred["raw_line_scores"] = raw_line_scores

        return pred


def log_double_softmax(scores, bin_score):
    b, m, n = scores.shape
    bin_ = bin_score[None, None, None]
    scores0 = tf.concat((scores, tf.tile(bin_, (b, m, 1))), axis=2)
    scores1 = tf.concat((scores, tf.tile(bin_, (b, 1, n))), axis=1)
    scores0 = tf.nn.log_softmax(scores0, 2)
    scores1 = tf.nn.log_softmax(scores1, 1)
    mean = (scores0[:, :, :n] + scores1[:, :m, :]) / 2
    p0 = scores0[:, :, -1]
    p1 = scores1[:, -1, :]
    return tf.concat((tf.concat((mean, p0[..., None]), axis=-1), tf.concat((p1, tf.zeros((b, 1))), axis=1)[None]), axis=1)


def gather_like_in_torch(indices, values):
    b_size, d = indices.shape
    batch_range = tf.range(b_size)[:, tf.newaxis]
    batch_range = tf.tile(batch_range, [1, d])
    combined_indices = tf.stack([batch_range, indices], axis=-1)
    return tf.gather_nd(values, combined_indices)
