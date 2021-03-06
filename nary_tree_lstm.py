import tensorflow as tf
from batch_tree import BatchTree, BatchTreeSample
import numpy as np
from sklearn import metrics
import collections


def calc_wt_init(fan_in=300):
    eps = 1.0 / np.sqrt(fan_in)
    return eps


class NarytreeLSTM(object):
    def __init__(self, config=None):
        self.config = config
        self.attention_module = AttentionModule(
            config.hidden_dim,
            config.matching_scheme,
            config.span_scheme
        )

        with tf.variable_scope("Embed", regularizer=None):

            if config.embeddings is not None:
                initializer = config.embeddings
            else:
                initializer = tf.random_uniform((config.num_emb, config.emb_dim))
            self.embedding = tf.Variable(initial_value=initializer, trainable=config.trainable_embeddings,
                                         dtype='float32')


        with tf.variable_scope("Node",
                               initializer=
                               # tf.ones_initializer(),
                               tf.contrib.layers.xavier_initializer(),
                               regularizer=tf.contrib.layers.l2_regularizer(self.config.reg)
                               ):

            self.U = tf.get_variable("U", [config.hidden_dim * config.degree , config.hidden_dim * (3 + config.degree)], initializer=tf.random_uniform_initializer(-calc_wt_init(config.hidden_dim),calc_wt_init(config.hidden_dim)))
            self.W = tf.get_variable("W", [config.emb_dim + config.hidden_dim * 2, 3 * config.hidden_dim], initializer=tf.random_uniform_initializer(-calc_wt_init(config.emb_dim),calc_wt_init(config.emb_dim)))
            self.Wf = tf.get_variable("Wf", [config.emb_dim + config.hidden_dim * 2, config.hidden_dim], initializer=tf.random_uniform_initializer(-calc_wt_init(config.emb_dim),calc_wt_init(config.emb_dim)))
            self.b = tf.get_variable("b", [config.hidden_dim*3], initializer=tf.random_uniform_initializer(-calc_wt_init(config.hidden_dim),calc_wt_init(config.hidden_dim)))#, regularizer=tf.contrib.layers.l2_regularizer(0.0))
            self.bf = tf.get_variable("bf", [config.hidden_dim], initializer=tf.random_uniform_initializer(-calc_wt_init(config.hidden_dim),calc_wt_init(config.hidden_dim)))#, regularizer=tf.contrib.layers.l2_regularizer(0.0))

            self.observables = tf.placeholder(tf.int32, shape=[None])
            self.flows = tf.placeholder(tf.int32, shape=[None])
            self.input_scatter = tf.placeholder(tf.int32, shape=[None])
            self.observables_indices = tf.placeholder(tf.int32, shape=[None])
            self.out_indices = tf.placeholder(tf.int32, shape=[None])
            self.scatter_out = tf.placeholder(tf.int32, shape=[None])
            self.scatter_in = tf.placeholder(tf.int32, shape=[None])
            self.scatter_in_indices = tf.placeholder(tf.int32, shape=[None])
            self.batch_size = tf.placeholder(tf.int32, shape=[])
            self.tree_height = tf.placeholder(tf.int32, shape=[])
            self.dropout = tf.placeholder(tf.float32, shape=[])
            self.child_scatter_indices = tf.placeholder(tf.int32, shape=[None])
            self.nodes_count = tf.placeholder(tf.int32, shape=[None])
            self.input_embed = tf.nn.embedding_lookup(self.embedding, self.observables)
            self.nodes_count_per_indice = tf.placeholder(tf.float32, shape=[None])

            self.sentences = tf.placeholder(tf.int32, shape=[None, None])
            self.lengths = tf.placeholder(tf.int32, shape=[None])
            self.tree_idxs = tf.placeholder(tf.int32, shape=[None])
            self.span_idxs = tf.placeholder(tf.int32, shape=[None, 2])

            self.attention = None

            self.training_variables = [self.U, self.W, self.b, self.bf]

            if config.trainable_embeddings:
                self.training_variables.append(self.embedding)

    def get_feed_dict(self, batch_sample, dropout = 1.0):
        return {
        self.observables : batch_sample.observables,
        self.flows : batch_sample.flows,
        self.input_scatter : batch_sample.input_scatter,
        self.observables_indices : batch_sample.observables_indices,
        self.out_indices: batch_sample.out_indices,
        self.tree_height: len(batch_sample.out_indices)-1,
        self.batch_size: batch_sample.flows[-1],#batch_sample.out_indices[-1] - batch_sample.out_indices[-2],
        self.scatter_out: batch_sample.scatter_out,
        self.scatter_in: batch_sample.scatter_in,
        self.scatter_in_indices: batch_sample.scatter_in_indices,
        self.child_scatter_indices: batch_sample.child_scatter_indices,
        self.nodes_count: batch_sample.nodes_count,
        self.dropout : dropout,
        self.nodes_count_per_indice : batch_sample.nodes_count_per_indice,
        self.sentences : batch_sample.sentences,
        self.lengths : batch_sample.sentence_lengths,
        self.tree_idxs : batch_sample.tree_idxs,
        self.span_idxs : batch_sample.span_idxs,
        }

    def get_output(self):
        nodes_h, _ = self.get_outputs()
        return nodes_h

    def get_output_unscattered(self):
        _, nodes_h_unscattered = self.get_outputs()
        return nodes_h_unscattered

    def get_attention(self):
        return self.attention

    def get_sentence_lstm_ouput(self, sentences, lengths, scope, is_bidirectional):
        sen_embedding = tf.nn.embedding_lookup(self.embedding, sentences)

        with tf.variable_scope(scope, reuse=True):
            cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.config.hidden_dim, reuse=tf.AUTO_REUSE)
            if is_bidirectional:
                cell_bw = tf.contrib.rnn.LayerNormBasicLSTMCell(self.config.hidden_dim, reuse=tf.AUTO_REUSE)
                bi_lstm_output, final_state = tf.nn.bidirectional_dynamic_rnn(cell, cell_bw, sen_embedding,
                                                                              sequence_length=lengths, dtype=tf.float32)
                return bi_lstm_output[0], bi_lstm_output[1]

            else:
                outputs, _ = tf.nn.dynamic_rnn(cell, sen_embedding, lengths, dtype=tf.float32)
                return outputs, outputs

    def get_outputs(self):

        with tf.variable_scope("Node", reuse=True):

            W = tf.get_variable("W", [self.config.emb_dim + self.config.hidden_dim * 2, 3 * self.config.hidden_dim])
            Wf = tf.get_variable("Wf", [self.config.emb_dim + self.config.hidden_dim * 2, self.config.hidden_dim])
            U = tf.get_variable("U", [self.config.hidden_dim * self.config.degree , self.config.hidden_dim * (3 + self.config.degree)])
            b = tf.get_variable("b", [3 * self.config.hidden_dim])
            bf = tf.get_variable("bf", [self.config.hidden_dim])

            nWf = tf.tile(Wf, [1, self.config.degree])
            W_ = tf.concat([nWf, W], axis=1)
            max_sentence_len = tf.reduce_max(self.lengths)

            attn_fw, attn_bw = self.get_sentence_lstm_ouput(self.sentences, self.lengths, "lstm_attn", is_bidirectional=True)

            nbf = tf.tile(bf, [self.config.degree])
            # nbf = tf.Print(nbf, [attn_src_l, tf.shape(attn_src_l)], "attn_src_l")

            nodes_h_scattered = tf.TensorArray(tf.float32, size=self.tree_height, clear_after_read=False)
            nodes_h = tf.TensorArray(tf.float32, size = self.tree_height, clear_after_read=False)
            nodes_c = tf.TensorArray(tf.float32, size = self.tree_height, clear_after_read=False)

            attn_arr = tf.TensorArray(tf.float32, size = self.tree_height, clear_after_read=False)

            const0f = tf.constant([0], dtype=tf.float32)
            idx_var = tf.constant(0, dtype=tf.int32)

            hidden_shape = tf.constant([-1, self.config.hidden_dim * self.config.degree], dtype=tf.int32)
            out_shape = tf.stack([-1,self.batch_size, self.config.hidden_dim], 0)

            def _recurrence(nodes_h, nodes_c, nodes_h_scattered, idx_var, attn_arr):
                out_ = tf.concat([nbf, b], axis=0)
                idx_var_dim1 = tf.expand_dims(idx_var, 0)
                prev_idx_var_dim1 = tf.expand_dims(idx_var-1, 0)

                observables_indice_begin, observables_indice_end = tf.split(tf.slice(self.observables_indices, idx_var_dim1, [2]), 2)
                observables_size = observables_indice_end - observables_indice_begin
                out_indice_begin, out_indice_end = tf.split(
                    tf.slice(self.out_indices, idx_var_dim1, [2]), 2)
                out_size = out_indice_end - out_indice_begin
                flow = tf.slice(self.flows, idx_var_dim1, [1])
                u_scatter_shape = tf.concat([flow, [self.config.hidden_dim * (3 + self.config.degree)]], axis=0)
                c_scatter_shape = tf.concat([flow, [self.config.hidden_dim * self.config.degree]],axis=0)

                def compute_indices():
                    prev_level_indice_begin, prev_level_indice_end = tf.split(
                        tf.slice(self.out_indices, prev_idx_var_dim1, [2]), 2)
                    prev_level_indice_size = prev_level_indice_end - prev_level_indice_begin
                    scatter_indice_begin, scatter_indice_end = tf.split(
                        tf.slice(self.scatter_in_indices, prev_idx_var_dim1, [2]), 2)
                    scatter_indice_size = scatter_indice_end - scatter_indice_begin
                    child_scatters = tf.slice(self.child_scatter_indices, prev_level_indice_begin, prev_level_indice_size)
                    child_scatters = tf.reshape(child_scatters, tf.concat([prev_level_indice_size, [-1]], 0))
                    return scatter_indice_begin, scatter_indice_size, child_scatters

                def hs_compute():
                    scatter_indice_begin, scatter_indice_size, child_scatters = compute_indices()

                    h = nodes_h.read(idx_var - 1)
                    hs = tf.scatter_nd(child_scatters,h,tf.shape(h), name=None)
                    hs = tf.reshape(hs, hidden_shape)
                    out = tf.matmul(hs, U)

                    scatters_in = tf.slice(self.scatter_in, scatter_indice_begin, scatter_indice_size)
                    scatters_in = tf.reshape(scatters_in, tf.concat([scatter_indice_size, [-1]], 0))
                    out = tf.scatter_nd(scatters_in, out, u_scatter_shape, name=None)
                    return out

                def cs_compute():
                    scatter_indice_begin, scatter_indice_size, child_scatters = compute_indices()

                    c = nodes_c.read(idx_var - 1)
                    cs = tf.scatter_nd(child_scatters, c, tf.shape(c), name=None)
                    cs = tf.reshape(cs, hidden_shape)

                    scatters_in = tf.slice(self.scatter_in, scatter_indice_begin, scatter_indice_size)
                    scatters_in = tf.reshape(scatters_in, tf.concat([scatter_indice_size, [-1]], 0))
                    cs = tf.scatter_nd(scatters_in, cs, c_scatter_shape, name=None)
                    return cs

                out_ += tf.cond(tf.less(0,idx_var),
                             lambda: hs_compute(),
                             lambda: const0f
                             )
                cs = tf.cond(tf.less(0,idx_var),
                             lambda: cs_compute(),
                             lambda: const0f
                             )

                observable = tf.squeeze(tf.slice(self.observables, observables_indice_begin, observables_size))

                input_embed = tf.reshape(tf.nn.embedding_lookup(self.embedding, observable),[-1,self.config.emb_dim])

                def compute_input():

                    input_embed_padded = tf.pad(input_embed, [[0, 0], [0, self.config.hidden_dim * 2]], "CONSTANT")
                    out = tf.matmul(input_embed_padded, W_)

                    input_scatter = tf.slice(self.input_scatter, observables_indice_begin, observables_size)
                    input_scatter = tf.reshape(input_scatter, tf.concat([observables_size, [-1]], 0))
                    out = tf.scatter_nd(input_scatter, out, u_scatter_shape, name=None)

                    return out

                computed_input_val = tf.cond(tf.less(0, tf.squeeze(observables_size)),
                               lambda: compute_input(),
                               lambda: const0f)

                out_ += computed_input_val

                def compute_attn_ctx(flat_src_l, flat_src_r):

                    scatter_indice_begin, scatter_indice_size, child_scatters = compute_indices()

                    child_hs_in_batch = nodes_h.read(idx_var - 1)
                    children_in_pairs = tf.scatter_nd(child_scatters, child_hs_in_batch, tf.shape(child_hs_in_batch),
                                                      name="child_batch_to_pairs")

                    # create sentence indices
                    level_indice_begin, level_indice_end = tf.split(tf.slice(self.out_indices, prev_idx_var_dim1, [2]),
                                                                    2)
                    level_indice_size = level_indice_end - level_indice_begin
                    tree_idx_in_batch = tf.slice(self.tree_idxs, level_indice_begin, level_indice_size)
                    tree_idx_in_pairs = tf.scatter_nd(child_scatters, tree_idx_in_batch, tf.shape(tree_idx_in_batch),
                                                      name="tree_idx_batch_to_pairs")

                    # create span indices
                    zero_expanded = tf.expand_dims(tf.constant(0), axis=0)
                    two_expanded = tf.expand_dims(tf.constant(2), axis=0)
                    span_in_batch = tf.slice(self.span_idxs, tf.concat([level_indice_begin, zero_expanded], axis=0),
                                             tf.concat([level_indice_size, two_expanded], axis=0))
                    span_idx_in_pairs = tf.scatter_nd(child_scatters, span_in_batch, tf.shape(span_in_batch),
                                                      name="span_idx_batch_to_pairs")

                    res_l, res_r = self.attention_module.compute(children_in_pairs,
                                                                 span_idx_in_pairs,
                                                                 tree_idx_in_pairs,
                                                                 flat_src_l, flat_src_r,
                                                                 batch_sentence_lengths=self.lengths,
                                                                 dropout=self.dropout)

                    (ctx_left, attn_weights_l), (ctx_right, attn_weights_r) = res_l, res_r

                    # combine context and put in input format, pre-padding size of original embedding dim
                    ctx_overall = tf.concat([ctx_left, ctx_right], axis=-1)
                    ctx_overall = tf.nn.dropout(ctx_overall, self.dropout)
                    ctx_overall = tf.pad(ctx_overall, [[0, 0], [self.config.emb_dim, 0]], "CONSTANT")

                    ctx_overall = tf.matmul(ctx_overall, W_)

                    attn_w_level = attn_weights_l + attn_weights_r

                    # project to input format of current level
                    scatters_in = tf.slice(self.scatter_in, scatter_indice_begin, scatter_indice_size)
                    scatters_in = tf.reshape(scatters_in, tf.concat([scatter_indice_size, [-1]], 0))
                    ctx_overall = tf.scatter_nd(scatters_in, ctx_overall, u_scatter_shape, name=None)

                    return ctx_overall, attn_w_level

                if self.config.attn_place == 'ROOT':
                    print("Root only attention ... ")
                    attention_cond = tf.equal(self.tree_height - 1, idx_var)
                elif self.config.attn_place == 'ALL':
                    print("All nodes attention ...")
                    attention_cond = tf.less(0, idx_var)
                elif self.config.attn_place == 'NONE':
                    print("No attention ...")
                    attention_cond = tf.greater(-1, idx_var)
                elif self.config.attn_place == '3':
                    print("Attention on 3-layer deep ...")
                    attention_cond = tf.less_equal(self.tree_height - 3, idx_var)
                else:
                    raise Exception("Unknown attention place")

                attn_ctx, attn_weights = tf.cond(attention_cond,
                                                 lambda: compute_attn_ctx(attn_bw, attn_fw),
                                                 lambda: (const0f, tf.zeros((1, max_sentence_len))))

                if self.config.attn_place != 'NONE':
                    out_ += attn_ctx
                    attn_arr = attn_arr.write(idx_var, attn_weights)

                v = tf.split(out_, 3 + self.config.degree, axis=1)

                vf = tf.sigmoid(tf.concat(v[:self.config.degree], axis=1))

                c = tf.cond(tf.less(0,idx_var),
                                         lambda: tf.multiply(tf.sigmoid(v[self.config.degree]),tf.tanh(v[self.config.degree+2])) + tf.reduce_sum(
                                             tf.stack(tf.split(tf.multiply(vf, cs), self.config.degree, axis=1)), axis=0),
                                         lambda: tf.multiply(tf.sigmoid(v[self.config.degree]),tf.tanh(v[self.config.degree+2]))
                                         )

                h = tf.multiply(tf.sigmoid(v[self.config.degree + 1]),tf.tanh(c))
                h = tf.nn.dropout(h, self.dropout)
                nodes_h = nodes_h.write(idx_var, h)
                nodes_c = nodes_c.write(idx_var, c)

                scatters = tf.reshape(tf.slice(self.scatter_out, out_indice_begin, out_size), tf.concat([out_size, [-1]], 0))

                node_count = tf.slice(self.nodes_count, idx_var_dim1, [1])
                scatter_out_lenght = node_count * self.batch_size
                scatter_out_shape = tf.stack([tf.squeeze(scatter_out_lenght), self.config.hidden_dim], 0)
                h = tf.reshape(tf.scatter_nd(scatters, h, scatter_out_shape, name=None), out_shape)
                nodes_h_scattered = nodes_h_scattered.write(idx_var, h)
                idx_var = tf.add(idx_var, 1)

                return nodes_h, nodes_c, nodes_h_scattered, idx_var, attn_arr

            loop_cond = lambda x, y, z, id, attn: tf.less(id, self.tree_height)

            loop_vars = [nodes_h, nodes_c, nodes_h_scattered, idx_var, attn_arr]
            nodes_h, nodes_c, nodes_h_scattered, idx_var, attn_arr = tf.while_loop(loop_cond, _recurrence, loop_vars,
                                                                                   parallel_iterations=1)

            if self.config.attn_place != 'NONE':
                self.attention = attn_arr.concat()
            else:
                self.attention = const0f

            return nodes_h_scattered.concat(), nodes_h


def restricted_softmax_on_sequence(logits, sentence_length, not_null_count):
    sequence_mask = tf.sequence_mask(tf.cast(not_null_count, dtype=tf.int32), sentence_length)
    binary_mask = tf.cast(sequence_mask, dtype=tf.float32)
    return restricted_softmax_on_mask(logits, binary_mask)


def restricted_softmax_on_sequence_range(logits, max_sen_length, inclusive_ranges):
    """
    :param logits: logits
    :param max_sen_length: max sentence length
    :param inclusive_ranges: [batch_size, 2] tensors, last dim corresponds to start and end.
    :return:
    """
    begin = inclusive_ranges[:, 0]
    end = inclusive_ranges[:, 1] + 1

    sequence_mask_begin = tf.sequence_mask(begin, max_sen_length)
    inverted_begin = tf.logical_not(sequence_mask_begin)
    sequence_mask_end = tf.sequence_mask(end, max_sen_length)

    range_mask = tf.logical_and(sequence_mask_end, inverted_begin)
    binary_mask = tf.cast(range_mask, dtype=tf.float32)

    return restricted_softmax_on_mask(logits, binary_mask)


def restricted_softmax_on_mask(logits, binary_mask):
    large_neg = (logits * 0) - 100000
    large_neg_on_empty = large_neg * (1 - binary_mask)
    return tf.nn.softmax(logits + large_neg_on_empty)


def build_mlp(
        input,
        output_size,
        drop_out,
        scope,
        n_layers=2,
        sizes=[64, 64],
        activation=tf.nn.relu,
        output_activation=None
        ):

    if len(sizes) != n_layers:
        raise Exception("Size of layers should correspond with layer size")

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        prev_layer = input
        for i in range(n_layers):
            prev_layer = tf.nn.dropout(prev_layer, drop_out)
            prev_layer = tf.layers.dense(inputs=prev_layer, units=sizes[i], activation=activation)

        prev_layer = tf.nn.dropout(prev_layer, drop_out)
        return tf.layers.dense(inputs=prev_layer, units=output_size, activation=output_activation)


class AttentionModule(object):

    def __init__(self, hidden_dim, matching_scheme, span_scheme):
        self.hidden_dim = hidden_dim
        self.matching_scheme = matching_scheme
        self.span_scheme = span_scheme

    def _split_left_right(self,
                          children_in_pairs,
                          span_idx_in_pairs,
                          tree_idx_in_pairs,
                          flat_src_l, flat_src_r,
                          batch_sentence_lengths):

        # create index to extract left and right children
        idx_range = tf.range(tf.shape(children_in_pairs)[0])
        even_odd_idx = tf.transpose(tf.reshape(idx_range, [-1, 2]))
        even_idx = even_odd_idx[0]
        odd_idx = even_odd_idx[1]
        child_l = tf.gather(children_in_pairs, even_idx)
        child_r = tf.gather(children_in_pairs, odd_idx)

        # extract the sentences for the level
        idx_left = tf.gather(tree_idx_in_pairs, even_idx)
        idx_right = tf.gather(tree_idx_in_pairs, odd_idx)
        flat_src_l = tf.gather(flat_src_l, idx_left)
        flat_src_r = tf.gather(flat_src_r, idx_right)

        if self.span_scheme == 'PARENT':
            print("Using parent span ..")
            span_left = tf.gather(span_idx_in_pairs, even_idx)
            span_right = tf.gather(span_idx_in_pairs, odd_idx)

            span_left_left = tf.expand_dims(span_left[:, 0], axis=-1)
            span_right_right = tf.expand_dims(span_right[:, 1], axis=-1)

            parent_span = tf.concat([span_left_left, span_right_right], axis=-1)

            span_left = parent_span
            span_right = parent_span

        elif self.span_scheme == 'SIBLING':
            print("Using sibling span ..")
            span_left = tf.gather(span_idx_in_pairs, even_idx)
            span_right = tf.gather(span_idx_in_pairs, odd_idx)

        elif self.span_scheme == 'ALL':
            print("Using whole sentence span ..")
            sentence_len = tf.gather(batch_sentence_lengths, idx_left)
            length_inclusive = tf.expand_dims(sentence_len - 1, axis=1)

            sentence_span = tf.pad(length_inclusive, tf.constant([[0, 0], [1, 0]]), 'CONSTANT')
            span_left = sentence_span
            span_right = sentence_span
        else:
            raise Exception("Unknown span scheme")

        return (child_l, child_r), (span_left, span_right), (flat_src_l, flat_src_r)

    def compute(self,
                children_in_pairs,
                span_idx_in_pairs,
                tree_idx_in_pairs,
                full_flat_src_l, full_flat_src_r,
                batch_sentence_lengths,
                dropout):

        (child_l, child_r), (span_left, span_right), (flat_src_l, flat_src_r) = self._split_left_right(
            children_in_pairs,
            span_idx_in_pairs,
            tree_idx_in_pairs,
            full_flat_src_l, full_flat_src_r,
            batch_sentence_lengths)

        # left child attending right span and vice versa
        (ctx_l, attn_ws_l) = self._compute_attentional_contexts(child_l, flat_src_l, span_right, is_left=True, dropout=dropout)
        (ctx_r, attn_ws_r) = self._compute_attentional_contexts(child_r, flat_src_r, span_left, is_left=False, dropout=dropout)

        return (ctx_l, attn_ws_l), (ctx_r, attn_ws_r)

    def _compute_attentional_contexts(self, h_child, flat_src, span, is_left, dropout):
        """
        Compute context and attention weights. Note that flat source here needs not be unique, because hidden state of child at the same level can come from the same tree.
        :param h_child: hidden child state in batch [num_child, hidden_dim]
        :param flat_src: flat sentence source where the order does correspond with the order of h_child
        :param span: start and begin of each sub-sentence of the flat source
        :param is_left: whether compute attention of left child
        :return: context and attention weights
        """

        scope = "attn_left" if is_left else "attn_right"

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

            h_child = tf.nn.dropout(h_child, dropout)
            flat_src = tf.nn.dropout(flat_src, dropout)

            h_child = tf.reshape(h_child, [-1, self.hidden_dim])
            h_child = tf.expand_dims(h_child, axis=1)

            if is_left:
                flat_src = tf.layers.dense(flat_src, self.hidden_dim, name="attn_src_proj_left",
                                           reuse=tf.AUTO_REUSE)
            else:
                flat_src = tf.layers.dense(flat_src, self.hidden_dim, name="attn_src_proj_right",
                                           reuse=tf.AUTO_REUSE)

            if self.matching_scheme == 'DOT_PROD':
                print("Using dot product for matching score...")
                matching_score = tf.reduce_sum(h_child * flat_src, axis=-1)
            elif self.matching_scheme == 'ADDITIVE':
                print("Using additive for matching score...")
                h_child = tf.layers.dense(h_child,
                                          self.config.hidden_dim,
                                          name="attn_child_proj",
                                          reuse=tf.AUTO_REUSE, use_bias=False)
                matching_score = tf.nn.dropout(tf.nn.relu(h_child + flat_src), dropout)
                matching_score = tf.squeeze(tf.layers.dense(matching_score, 1, name="attn_score", reuse=tf.AUTO_REUSE),
                                            axis=-1)
            elif self.matching_scheme == 'MLP':
                print("Building mlp for matching score...")
                h_child = tf.tile(h_child, [1, tf.shape(flat_src)[1], 1])
                matching_score = build_mlp(
                    input=tf.concat([h_child, flat_src], axis=-1),
                    output_size=1,
                    drop_out=1,
                    scope="matching_mlp",
                    n_layers=2,
                    sizes=[4, 4],
                    output_activation=tf.nn.relu)
                matching_score = tf.squeeze(matching_score, axis=-1)
            else:
                raise Exception("Matching func {} is not known".format(self.matching_scheme))

            attn_ws = restricted_softmax_on_sequence_range(matching_score, tf.shape(matching_score)[1], span)
            context = tf.reduce_sum(flat_src * tf.expand_dims(attn_ws, axis=-1), axis=1)

            # scaling adjusting for different size
            span_size = (span[:, 1] - span[:, 0]) + 1
            span_size_dim = tf.cast(tf.expand_dims(span_size, axis=1), dtype=tf.float32)
            context = context * (span_size_dim / 40)

            return context, attn_ws


class SoftMaxNarytreeLSTM(object):

    def __init__(self, config):
        def calc_wt_init(self, fan_in=300):
            eps = 1.0 / np.sqrt(fan_in)
            return eps
        self.config = config
        with tf.variable_scope("Predictor",
                               initializer=
                               tf.contrib.layers.xavier_initializer(),
                               regularizer=tf.contrib.layers.l2_regularizer(self.config.reg)
                               ):
            self.tree_lstm = NarytreeLSTM(config)
            self.W = tf.get_variable("W", [config.hidden_dim, config.num_labels], initializer=tf.random_uniform_initializer(-calc_wt_init(config.hidden_dim),calc_wt_init(config.hidden_dim)))
            self.b = tf.get_variable("b", [config.num_labels], initializer=tf.random_uniform_initializer(-calc_wt_init(config.hidden_dim),calc_wt_init(config.hidden_dim)))#, regularizer=tf.contrib.layers.l2_regularizer(0.0))
            self.labels = tf.placeholder(tf.int32, [None], name="labels")
            self.training_variables = [self.W, self.b] + self.tree_lstm.training_variables
            self.optimizer = tf.train.AdagradOptimizer(self.config.lr)
            self.embed_optimizer = tf.train.AdagradOptimizer(self.config.emb_lr)
            self.loss = self.get_loss()
            if config.trainable_embeddings:
                tvars = tf.trainable_variables()
                # embedding was assumed to be at the last position in the original code in self.training_variables
                # here we make sure we can identify it for a distinct gradient flow
                tvars.remove(self.training_variables[-1])
                tvars.append(self.training_variables[-1])
                self.gv = zip(tf.gradients(self.loss, tvars), tvars)
                self.opt = self.optimizer.apply_gradients(self.gv[:-1])
                self.embed_opt = self.embed_optimizer.apply_gradients(self.gv[-1:])
            else:
                self.opt = self.optimizer.apply_gradients(self.gv)
                self.embed_opt = tf.no_op()

            self.output = self.get_root_output()
            self.attention = self.tree_lstm.attention

    def get_root_output(self):
        nodes_h = self.tree_lstm.get_output_unscattered()
        roots_h = nodes_h.read(nodes_h.size()-1)
        out = tf.matmul(roots_h, self.W) + self.b
        return out

    def get_output(self):
        return self.output

    def get_attention(self):
        return self.attention

    def get_loss(self):
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        regpart = tf.add_n(reg_losses)
        #regpart = tf.Print(regpart, [regpart])
        h = self.tree_lstm.get_output_unscattered().concat()
        out = tf.matmul(h, self.W) + self.b
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=out)
        return tf.reduce_sum(tf.divide(loss, tf.to_float(self.tree_lstm.batch_size))) + regpart

    def train(self, batch_tree, batch_labels, session):

        feed_dict = {self.labels: batch_tree.labels}
        feed_dict.update(self.tree_lstm.get_feed_dict(batch_tree, self.config.dropout))
        ce,_,_ = session.run([self.loss, self.opt, self.embed_opt], feed_dict=feed_dict)
        #v = session.run([self.output], feed_dict=feed_dict)
        #print("cross_entropy " + str(ce))
        return ce
        #print v

    def train_epoch(self, data, session):
        #from random import shuffle
        #shuffle(data)
        total_error = 0.0
        for batch in data:
            total_error += self.train(batch[0], batch[1], session)
        print 'average error :', total_error/len(data)
        return total_error/len(data)

    def test(self, data, session):
        ys_true = collections.deque([])
        ys_pred = collections.deque([])
        for batch in data:
            y_pred = tf.argmax(self.get_output(), 1)
            y_true = self.labels
            feed_dict = {self.labels: batch[0].root_labels}
            feed_dict.update(self.tree_lstm.get_feed_dict(batch[0]))
            y_pred, y_true = session.run([y_pred, y_true], feed_dict=feed_dict)
            ys_true += y_true.tolist()
            ys_pred += y_pred.tolist()
        ys_true = list(ys_true)
        ys_pred = list(ys_pred)
        score = metrics.accuracy_score(ys_true, ys_pred)
        print "Accuracy", score
        #print "Recall", metrics.recall_score(ys_true, ys_pred)
        #print "f1_score", metrics.f1_score(ys_true, ys_pred)
        print "confusion_matrix"
        print metrics.confusion_matrix(ys_true, ys_pred)
        return score


# def test_lstm_model():
#     class Config(object):
#         num_emb = 10
#         emb_dim = 3
#         hidden_dim = 4
#         output_dim = None
#         degree = 2
#         num_epochs = 3
#         early_stopping = 2
#         dropout = 0.5
#         lr = 1.0
#         emb_lr = 0.1
#         reg = 0.0001
#         fine_grained = False
#         trainable_embeddings = False
#         embeddings = None
#         batch_size=7
#
#     tree = BatchTree.empty_tree()
#     tree.root.add_sample(-1, 1)
#     tree.root.expand_or_add_child(-1, 1, 0)
#     tree.root.expand_or_add_child(1, 1, 1)
#     tree.root.children[0].expand_or_add_child(1, 0, 0)
#     tree.root.children[0].expand_or_add_child(1, 0,  1)
#
#     tree.root.add_sample(-1, 1)
#     tree.root.expand_or_add_child(2, 1, 0)
#     tree.root.expand_or_add_child(2, 1, 1)
#
#     tree.root.add_sample(-1, 1)
#     tree.root.expand_or_add_child(-1, 1, 0)
#     tree.root.expand_or_add_child(3, 1, 1)
#     tree.root.children[0].expand_or_add_child(3, 0, 0)
#     tree.root.children[0].expand_or_add_child(3, 0, 1)
#
#     sample = BatchTreeSample(tree)
#
#     model = NarytreeLSTM(Config())
#     sess = tf.InteractiveSession()
#     tf.global_variables_initializer().run()
#     v = sess.run(model.get_output(),feed_dict=model.get_feed_dict(sample))
#     print(v)
#     return 0


def test_softmax_model():
    class Config(object):
        num_emb = 10
        emb_dim = 500
        hidden_dim = 200
        output_dim = None
        degree = 2
        num_epochs = 3
        early_stopping = 2
        dropout = 0.5
        lr = 1.0
        emb_lr = 0.1
        reg = 0.0001
        fine_grained = False
        trainable_embeddings = True
        num_labels = 301
        embeddings = None

    tree = BatchTree.empty_tree()
    tree_0 = 0
    tree.root.add_sample(-1, 100, tree_0, [0, 1])
    tree.root.expand_or_add_child(0, 10, 0, tree_0, [0, 0])
    tree.root.expand_or_add_child(0, 20, 1, tree_0, [1, 1])

    tree_1 = 1
    tree.root.add_sample(-1, 200, tree_1, [0, 2])
    tree.root.expand_or_add_child(-1, 30, 0, tree_1, [0, 1])
    tree.root.expand_or_add_child(-1, 40, 1, tree_1, [2, 2])
    tree.root.children[0].expand_or_add_child(0, 1, 0, tree_1, [0, 0])
    tree.root.children[0].expand_or_add_child(0, 2, 1, tree_1, [1, 1])

    tree_2 = 2
    tree.root.add_sample(-1, 300, tree_2, [0, 3])
    tree.root.expand_or_add_child(-1, 50, 0, tree_2, [0, 1])
    tree.root.expand_or_add_child(-1, 60, 1, tree_2, [2, 3])
    tree.root.children[0].expand_or_add_child(0, 3, 0, tree_2, [0, 0])
    tree.root.children[0].expand_or_add_child(0, 4, 1, tree_2, [1, 1])
    tree.root.children[1].expand_or_add_child(0, 5, 0, tree_2, [2, 2])
    tree.root.children[1].expand_or_add_child(0, 6, 1, tree_2, [3, 3])

    batch_sample = BatchTreeSample(tree)

    all_sen = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0]])
    lens = [2, 2, 4]
    batch_sample.add_batch_sentences(all_sen, lens)

    observables, flows, mask, scatter_out, scatter_in, scatter_in_indices, labels, observables_indices, out_indices, childs_transpose_scatter, nodes_count, nodes_count_per_indice, tree_idx = tree.build_batch_tree_sample()
    print observables, "observables"
    print observables_indices, "observables_indices"
    print flows, "flows"
    print mask, "input_scatter"
    print scatter_out, "scatter_out"
    print scatter_in, "scatter_in"
    print scatter_in_indices, "scatter_in_indices"
    print labels, "labels"
    print out_indices, "out_indices"
    print childs_transpose_scatter, "childs_transpose_scatter"
    print nodes_count, "nodes_count"
    print nodes_count_per_indice, "nodes_count_per_indice"
    print tree_idx, "tree idxs"

    labels = np.array([0,1,0,1,0])

    model = SoftMaxNarytreeLSTM(Config())
    sess = tf.InteractiveSession()
    summarywriter = tf.summary.FileWriter('/tmp/tensortest', graph=sess.graph)
    tf.global_variables_initializer().run()
    sample = [(batch_sample, labels)]
    for i in range(100):
        model.train(batch_sample, labels, sess)
        model.test(sample, sess)
    return 0


if __name__ == '__main__':
    test_softmax_model()
    #test_lstm_model()