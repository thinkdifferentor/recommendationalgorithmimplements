import functools
import numpy
import tensorflow as tf
import toolz
import time
from tqdm import tqdm
from evaluator import RecallEvaluator
from sampler import WarpSampler
from dateset import citeulike, split_data


def doublewrap(function):
    """
    A decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator

@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the compute graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


class CML(object):
    def __init__(self,
                 n_users,
                 n_items,
                 embed_dim=20,
                 features=None,
                 margin=1.5,
                 master_learning_rate=0.1,
                 clip_norm=1.0,
                 hidden_layer_dim=128,
                 dropout_rate=0.2,
                 feature_l2_reg=0.1,
                 feature_projection_scaling_factor=0.5,
                 use_rank_weight=True,
                 use_cov_loss=True,
                 cov_loss_weight=0.1
                 ):
        """
        :param n_users: number of users i.e. |U|
        :param n_items: number of items i.e. |V|
        :param embed_dim: embedding size i.e. K (default 20)
        :param features: (optional) the feature vectors of items, shape: (|V|, N_Features).
               Set it to None will disable feature loss(default: None)
        :param margin: hinge loss threshold i.e. z
        :param master_learning_rate: master learning rate for AdaGrad
        :param clip_norm: clip norm threshold (default 1.0)
        :param hidden_layer_dim: the size of feature projector's hidden layer (default: 128)
        :param dropout_rate: the dropout rate between the hidden layer to final feature projection layer
        :param feature_l2_reg: feature loss weight
        :param feature_projection_scaling_factor: scale the feature projection before compute l2 loss. Ideally,
               the scaled feature projection should be mostly within the clip_norm
        :param use_rank_weight: whether to use rank weight
        :param use_cov_loss: use covariance loss to discourage redundancy in the user/item embedding
        """
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.clip_norm = clip_norm
        self.margin = margin
        if features is not None:
            self.features = tf.constant(features, dtype=tf.float32)
        else:
            self.features = None

        self.master_learning_rate = master_learning_rate
        self.hidden_layer_dim = hidden_layer_dim
        self.dropout_rate = dropout_rate
        self.feature_l2_reg = feature_l2_reg
        self.feature_projection_scaling_factor = feature_projection_scaling_factor
        self.use_rank_weight = use_rank_weight
        self.use_cov_loss = use_cov_loss
        self.cov_loss_weight = cov_loss_weight

        self.user_positive_items_pairs = tf.placeholder(tf.int32, [None, 2])
        self.negative_samples = tf.placeholder(tf.int32, [None, None])
        self.score_user_ids = tf.placeholder(tf.int32, [None])

        self.user_embeddings
        self.item_embeddings
        self.embedding_loss
        self.feature_loss
        self.loss
        self.optimize

    @define_scope
    def user_embeddings(self):
        return tf.Variable(tf.random_normal([self.n_users, self.embed_dim],
                                            stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32))

    @define_scope
    def item_embeddings(self):
        return tf.Variable(tf.random_normal([self.n_items, self.embed_dim],
                                            stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32))

    @define_scope
    def mlp_layer_1(self):
        return tf.layers.dense(inputs=self.features,
                               units=self.hidden_layer_dim,
                               activation=tf.nn.relu, name="mlp_layer_1")

    @define_scope
    def mlp_layer_2(self):
        dropout = tf.layers.dropout(inputs=self.mlp_layer_1, rate=self.dropout_rate)
        return tf.layers.dense(inputs=dropout, units=self.embed_dim, name="mlp_layer_2")

    @define_scope
    def feature_projection(self):
        """
        :return: the projection of the feature vectors to the user-item embedding
        """
        # feature loss
        if self.features is not None:
            # fully-connected layer
            output = self.mlp_layer_2 * self.feature_projection_scaling_factor
            # projection to the embedding
            return tf.clip_by_norm(output, self.clip_norm, axes=[1], name="feature_projection")

    @define_scope
    def feature_loss(self):
        """
        :return: the l2 loss of the distance between items' their embedding and their feature projection
        """
        loss = tf.constant(0, dtype=tf.float32)
        if self.feature_projection is not None:
            # the distance between feature projection and the item's actual location in the embedding
            feature_distance = tf.reduce_sum(tf.squared_difference(
                self.item_embeddings,
                self.feature_projection), 1)

            # apply regularization weight
            loss += tf.reduce_sum(feature_distance, name="feature_loss") * self.feature_l2_reg
        return loss

    @define_scope
    def covariance_loss(self):
        X = tf.concat((self.item_embeddings, self.user_embeddings), 0)
        n_rows = tf.cast(tf.shape(X)[0], tf.float32)
        X = X - (tf.reduce_mean(X, axis=0))
        cov = tf.matmul(X, X, transpose_a=True) / n_rows
        return tf.reduce_sum(tf.matrix_set_diag(cov, tf.zeros(self.embed_dim, tf.float32))) * self.cov_loss_weight

    @define_scope
    def embedding_loss(self):
        """
        :return: the distance metric loss
        """
        # Let
        # N = batch size,
        # K = embedding size,
        # W = number of negative samples per a user-positive-item pair

        # user embedding (N, K)
        users = tf.nn.embedding_lookup(self.user_embeddings,
                                       self.user_positive_items_pairs[:, 0],
                                       name="users")
        # positive item embedding (N, K)
        pos_items = tf.nn.embedding_lookup(self.item_embeddings, self.user_positive_items_pairs[:, 1],
                                           name="pos_items")
        # positive item to user distance (N)
        pos_distances = tf.reduce_sum(tf.squared_difference(users, pos_items), 1, name="pos_distances")

        # negative item embedding (N, K, W)
        neg_items = tf.transpose(tf.nn.embedding_lookup(self.item_embeddings, self.negative_samples),
                                 (0, 2, 1), name="neg_items")
        # distance to negative items (N x W)
        distance_to_neg_items = tf.reduce_sum(tf.squared_difference(tf.expand_dims(users, -1), neg_items), 1,
                                              name="distance_to_neg_items")

        # best negative item (among W negative samples) their distance to the user embedding (N)
        closest_negative_item_distances = tf.reduce_min(distance_to_neg_items, 1, name="closest_negative_distances")

        # compute hinge loss (N)
        loss_per_pair = tf.maximum(pos_distances - closest_negative_item_distances + self.margin, 0, name="pair_loss")

        # This scheme penalizes a positive item at a lower rank much more heavily than one at the top.
        if self.use_rank_weight:
            # indicator matrix for impostors (N x W)
            impostors = (tf.expand_dims(pos_distances, -1) - distance_to_neg_items + self.margin) > 0
            # approximate the rank of positive item by (number of impostor / W per user-positive pair) ???
            rank = tf.reduce_mean(tf.cast(impostors, dtype=tf.float32), 1, name="rank_weight") * self.n_items
            # apply rank weight (N)
            loss_per_pair *= tf.log(rank + 1)

        # the embedding loss
        loss = tf.reduce_sum(loss_per_pair, name="loss")

        return loss

    @define_scope
    def loss(self):
        """
        :return: the total loss = embedding loss + feature loss
        """
        loss = self.embedding_loss + self.feature_loss
        if self.use_cov_loss:
            loss += self.covariance_loss
        return loss

    @define_scope
    def clip_by_norm_op(self):
        return [tf.assign(self.user_embeddings, tf.clip_by_norm(self.user_embeddings, self.clip_norm, axes=[1])),
                tf.assign(self.item_embeddings, tf.clip_by_norm(self.item_embeddings, self.clip_norm, axes=[1]))]

    @define_scope
    def optimize(self):
        # have two separate learning rates. The first one for user/item embedding is un-normalized.
        # The second one for feature projector NN is normalized by the number of items.
        gds = []
        gds.append(tf.train
                   .AdagradOptimizer(self.master_learning_rate)
                   .minimize(self.loss, var_list=[self.user_embeddings, self.item_embeddings]))
        if self.feature_projection is not None:
            gds.append(tf.train
                       .AdagradOptimizer(self.master_learning_rate)
                       .minimize(self.feature_loss / self.n_items))

        with tf.control_dependencies(gds):
            return gds + [self.clip_by_norm_op]

    @define_scope
    def item_scores(self):
        # (N_USER_IDS, 1, K)
        user = tf.expand_dims(tf.nn.embedding_lookup(self.user_embeddings, self.score_user_ids), 1)
        # (1, N_ITEM, K)
        item = tf.expand_dims(self.item_embeddings, 0)
        # score = minus distance (N_USER, N_ITEM)
        return -tf.reduce_sum(tf.squared_difference(user, item), 2, name="scores")


# Hyper paramters
BATCH_SIZE = 10000
N_NEGATIVE = 20
EVALUATION_EVERY_N_BATCHES = 30
EMBED_DIM = 100
EARLY_STOPPING_ROUND = 100


def optimize(model, sampler, train, valid):
    """
    Optimize the model. TODO: implement early-stopping
    :param model: model to optimize
    :param sampler: mini-batch sampler
    :param train: train user-item matrix
    :param valid: validation user-item matrix
    :return: None
    """
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    if model.feature_projection is not None:
        # initialize item embedding with feature projection
        sess.run(tf.assign(model.item_embeddings, model.feature_projection))

    # sample some users to calculate recall validation
    valid_users = numpy.random.choice(list(set(valid.nonzero()[0])), size=1000, replace=False)

    # Log
    log_out_file =  'Log/BATCH_SIZE%d_N_NEGATIVE%d_EMBED_DIM%d_EARLY_STOPPING_ROUND%d_%s.log' \
        %(BATCH_SIZE, N_NEGATIVE,EMBED_DIM, EARLY_STOPPING_ROUND, time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
    log =  open(log_out_file, mode = "a+")

    cur_recall, best_recall, round = 0.0, 0.0, 0
    while True:
        # Evaluation step.
        # create evaluator on validation set
        validation_recall = RecallEvaluator(model, train, valid)
        # compute recall on validate set
        valid_recalls = []
        # compute recall in chunks to utilize speedup provided by Tensorflow
        for user_chunk in toolz.partition_all(100, valid_users):
            valid_recalls.extend([validation_recall.eval(sess, user_chunk)])
        cur_recall = numpy.mean(valid_recalls)
        print("Round {} recall on (sampled) validation set: {}\n".format(round, cur_recall))
        log.write("Round {} recall on (sampled) validation set: {}\n".format(round, cur_recall))

        # update the best recall.
        if cur_recall > best_recall:
            best_recall = cur_recall
        
        # *early stopping based on validation recall
        if EARLY_STOPPING_ROUND == round:
            break

        # Train model.
        losses = []
        # run n mini-batches
        for _ in tqdm(range(EVALUATION_EVERY_N_BATCHES), desc="Optimizing..."):
            # Get a batch of train set. Not the total examples.
            # The train data unit is a batch of train set, 
            # include batch size of (user-positive-item, negetive items)
            user_pos, neg = sampler.next_batch()
            _, loss = sess.run((model.optimize, model.loss),
                               {model.user_positive_items_pairs: user_pos,
                                model.negative_samples: neg})
            losses.append(loss)
        round += 1
        print("Round {} training loss {}\n".format(round, numpy.mean(losses)))
        log.write("Round {} training loss {}\n".format(round, numpy.mean(losses)))

    print("The best recall is {}\n".format(best_recall))
    log.write("The best recall is {}\n".format(best_recall))
    log.close()


if __name__ == '__main__':
    # get user-item matrix
    user_item_matrix, features = citeulike(tag_occurence_thres=5)
    n_users, n_items = user_item_matrix.shape
    
    # make feature as dense matrix
    dense_features = features.toarray() + 1E-10
    
    # get train/valid/test user-item matrices
    train, valid, test = split_data(user_item_matrix)
    
    # create warp sampler
    sampler = WarpSampler(train, batch_size=BATCH_SIZE, n_negative=N_NEGATIVE, check_negative=True)

    # WITHOUT features
    # Train a user-item joint embedding, where the items a user likes will be pulled closer to this users.
    # Once the embedding is trained, the recommendations are made by finding the k-Nearest-Neighbor to each user.
    model = CML(n_users,
                n_items,
                # set features to None to disable feature projection
                features=None,
                # size of embedding
                embed_dim=EMBED_DIM,
                # the size of hinge loss margin. test value 1.9/0.5
                margin=0.5,
                # clip the embedding so that their norm <= clip_norm
                clip_norm=1,
                # learning rate for AdaGrad
                master_learning_rate=0.1,

                # whether to enable rank weight. If True, the loss will be scaled by the estimated
                # log-rank of the positive items. If False, no weight will be applied.
                # This is particularly useful to speed up the training for large item set.
                # Weston, Jason, Samy Bengio, and Nicolas Usunier.
                # "Wsabie: Scaling up to large vocabulary image annotation." IJCAI. Vol. 11. 2011.
                use_rank_weight=True,

                # whether to enable covariance regularization to encourage efficient use of the vector space.
                # More useful when the size of embedding is smaller (e.g. < 20 ).
                use_cov_loss=False,

                # weight of the cov_loss
                cov_loss_weight=1
            )
    # optimize(model, sampler, train, valid)

    # WITH features
    # In this case, we additionally train a feature projector to project raw item features into the
    # embedding. The projection serves as "a prior" to inform the item's potential location in the embedding.
    # We use a two fully-connected layers NN as our feature projector. (This model is much more computation intensive.
    # A GPU machine is recommended)
    model = CML(n_users,
                n_items,
                # enable feature projection
                features=dense_features,
                embed_dim=EMBED_DIM,
                # the size of hinge loss margin. test value 2.0/0.5
                margin=2.0,
                clip_norm=1.1,
                master_learning_rate=0.1,
                # the size of the hidden layer in the feature projector NN
                hidden_layer_dim=512,
                # dropout rate between hidden layer and output layer in the feature projector NN
                dropout_rate=0.3,
                # scale the output of the NN so that the magnitude of the NN output is closer to the item embedding
                feature_projection_scaling_factor=1,
                # the penalty to the distance between projection and item's actual location in the embedding
                # tune this to adjust how much the embedding should be biased towards the item features.
                feature_l2_reg=0.1,

                # whether to enable rank weight. If True, the loss will be scaled by the estimated
                # log-rank of the positive items. If False, no weight will be applied.
                # This is particularly useful to speed up the training for large item set.
                # Weston, Jason, Samy Bengio, and Nicolas Usunier.
                # "Wsabie: Scaling up to large vocabulary image annotation." IJCAI. Vol. 11. 2011.
                use_rank_weight=True,

                # whether to enable covariance regularization to encourage efficient use of the vector space.
                # More useful when the size of embedding is smaller (e.g. < 20 ).
                use_cov_loss=False,

                # weight of the cov_loss
                cov_loss_weight=1
            )
    optimize(model, sampler, train, valid)
