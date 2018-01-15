import tensorflow as tf

from ..utils.cyclegan_hook import CycleGANHook
from ..utils.model_utils import *
from ..registry import register
from .discriminators.registry import _DISCRIMINATORS
from .transformations.registry import _TRANSFORMATIONS


@register("cycle_gan")
def get_cycle_gan(params, lr):
  """Callable model function compatible with Experiment API.

  Args:
    params: a HParams object containing values for fields:
      F: A network mapping domain X to domain Y
      G: A network mapping domain Y to domain X
      discriminator_X: A binary classifier that distinguish elements in X
      discriminator_Y: A binary classifier that distinguish elements in Y
  """

  def cycle_gan_text(F, G, discrim_X, discrim_Y, X, Y, X_groundtruth,
                     Y_groundtruth, past_Xs, past_Ys, mode):
    """Run CycleGAN on text."""
    X_dist = tf.one_hot(X, depth=params.vocab_size)
    Y_dist = tf.one_hot(Y, depth=params.vocab_size)
    X_groundtruth_dist = tf.one_hot(X_groundtruth, depth=params.vocab_size)
    Y_groundtruth_dist = tf.one_hot(Y_groundtruth, depth=params.vocab_size)

    X_mask = tf.where(tf.equal(X, 0), tf.zeros_like(X), tf.ones_like(X))
    X_mask = tf.tile(tf.expand_dims(X_mask, -1), [1, 1, params.vocab_size])
    Y_mask = tf.where(tf.equal(Y, 0), tf.zeros_like(Y), tf.ones_like(Y))
    Y_mask = tf.tile(tf.expand_dims(Y_mask, -1), [1, 1, params.vocab_size])

    if params.use_embeddings:
      X = embed_inputs(X, params)
      Y = embed_inputs(Y, params, reuse=True)
    else:
      X = X_dist
      Y = Y_dist

    with tf.variable_scope("transforms") as scope:
      Y_hat, generator_F_weights = collect_vars(lambda: F(X, params.F, params))
      X_hat, generator_G_weights = collect_vars(lambda: G(Y, params.G, params))

      scope.reuse_variables()

      if params.use_embeddings:
        Y_hat_emb = softmax_to_embedding(Y_hat, params)
        X_reconstruction = G(Y_hat_emb, params.G, params)
        X_hat_emb = softmax_to_embedding(X_hat, params)
        Y_reconstruction = F(X_hat_emb, params.F, params)
      else:
        X_reconstruction = G(Y_hat, params.G, params)
        Y_reconstruction = F(X_hat, params.F, params)

      # Ground truth loss logging. A metric for performance
      # ======================================================================
      X_groundtruth_loss = tf.reduce_mean(
          tf.abs(X_hat - X_groundtruth_dist) * tf.to_float(Y_mask))
      Y_groundtruth_loss = tf.reduce_mean(
          tf.abs(Y_hat - Y_groundtruth_dist) * tf.to_float(X_mask))
      X_groundtruth_acc = groundtruth_accuracy(
          tf.argmax(X_hat, axis=-1), X_groundtruth, Y_mask[:, :, 0])
      Y_groundtruth_acc = groundtruth_accuracy(
          tf.argmax(
              Y_hat, axis=-1), Y_groundtruth, X_mask[:, :, 0])
      tf.summary.scalar("X_groundtruth_loss", X_groundtruth_loss)
      tf.summary.scalar("Y_groundtruth_loss", Y_groundtruth_loss)
      tf.summary.scalar("X_groundtruth_acc", X_groundtruth_acc)
      tf.summary.scalar("Y_groundtruth_acc", Y_groundtruth_acc)
      # ======================================================================
      # Inspect mappings of the batch sequences.
      # ======================================================================
      lookup_table = construct_vocab_lookup_table(params.vocab)
      X_text = lookup_table.lookup(tf.argmax(X_dist[:3, :10, :], axis=-1))
      Y_text = lookup_table.lookup(tf.argmax(Y_dist[:3, :10, :], axis=-1))
      X_hat_text = lookup_table.lookup(tf.argmax(X_hat[:3, :10, :], axis=-1))
      Y_hat_text = lookup_table.lookup(tf.argmax(Y_hat[:3, :10, :], axis=-1))
      X_reconstruction_text = lookup_table.lookup(
          tf.argmax(X_reconstruction[:3, :10, :], axis=-1))
      Y_reconstruction_text = lookup_table.lookup(
          tf.argmax(Y_reconstruction[:3, :10, :], axis=-1))
      X_out_text = tf.string_join(
          [X_text, "->", Y_hat_text, "->", X_reconstruction_text])
      tf.summary.text("X", X_out_text)
      Y_out_text = tf.string_join(
          [Y_text, "->", X_hat_text, "->", Y_reconstruction_text])
      tf.summary.text("Y", Y_out_text)
      X_gt_text = lookup_table.lookup(X_groundtruth[:3, :10])
      Y_gt_text = lookup_table.lookup(Y_groundtruth[:3, :10])
      X_out_text = tf.string_join([X_gt_text, "|", X_hat_text])
      tf.summary.text("X_gt/X_actual", X_out_text)
      Y_out_text = tf.string_join([Y_gt_text, "|", Y_hat_text])
      tf.summary.text("Y_gt/Y_actual", Y_out_text)
      # ======================================================================

      # Text logging
      #log_text(F, G, params)

    with tf.variable_scope("discriminators") as scope:

      def discrim_loss(X, Y, train_towards_true, do_collect_vars=False):
        """Returns the discriminator loss.

        Args:
          X: Tensor from X domain to discriminate.
          Y: Tensor from Y domain to discriminate.
          train_towards_true: Bool. Train network to have discriminator recognize
            X and Y as true data samples.
          do_collect_vars: Bool. Indicates whether to return the newly constructed
            variables.
        """
        X_discrim, discrim_X_vars = collect_vars(
            lambda: discrim_X(X, params.discriminator_X, params))
        Y_discrim, discrim_Y_vars = collect_vars(
            lambda: discrim_Y(Y, params.discriminator_Y, params))

        if params.discrim_loss == "full_wgan":
          X_discrim_loss = -tf.reduce_mean(X_discrim)
          Y_discrim_loss = -tf.reduce_mean(Y_discrim)
          if train_towards_true:
            X_discrim_loss *= -1
            Y_discrim_loss *= -1
        elif params.discrim_loss == "log":
          X_discrim_loss = tf.log(X_discrim)
          Y_discrim_loss = tf.log(Y_discrim)
          if train_towards_true:
            X_discrim_loss *= -1
            Y_discrim_loss *= -1
          X_discrim_loss = tf.reduce_mean(X_discrim_loss)
          Y_discrim_loss = tf.reduce_mean(Y_discrim_loss)
        else:
          if train_towards_true:
            X_discrim_loss = tf.reduce_mean(
                tf.squared_difference(X_discrim, 1))
            Y_discrim_loss = tf.reduce_mean(
                tf.squared_difference(Y_discrim, 1))
          else:
            X_discrim_loss = tf.reduce_mean(X_discrim**2)
            Y_discrim_loss = tf.reduce_mean(Y_discrim**2)

        if do_collect_vars:
          return X_discrim_loss, Y_discrim_loss, discrim_X_vars, discrim_Y_vars
        else:
          return X_discrim_loss, Y_discrim_loss

      if params.use_embeddings:
        (X_hat_discrim_loss, Y_hat_discrim_loss, discrim_X_weights,
         discrim_Y_weights) = discrim_loss(X_hat_emb, Y_hat_emb, True, True)
      else:
        (X_hat_discrim_loss, Y_hat_discrim_loss, discrim_X_weights,
         discrim_Y_weights) = discrim_loss(X_hat, Y_hat, True, True)

      scope.reuse_variables()
      X_true_discrim_loss, Y_true_discrim_loss = discrim_loss(X, Y, True)

      # We only discriminate on the past example pool if we're training
      if mode == tf.contrib.learn.ModeKeys.TRAIN:
        if params.use_embeddings:
          past_X_emb = softmax_to_embedding(past_Xs, params)
          past_Y_emb = softmax_to_embedding(past_Ys, params)
          X_past_discrim_loss, Y_past_discrim_loss = discrim_loss(
              past_X_emb, past_Y_emb, False)
        else:
          X_past_discrim_loss, Y_past_discrim_loss = discrim_loss(
              past_Xs, past_Ys, False)
      else:
        X_past_discrim_loss = X_true_discrim_loss
        Y_past_discrim_loss = Y_true_discrim_loss

      if params.use_wasserstein:
        X_wass_penalty = wasserstein_penalty(discrim_X, X_dist, X_hat, params,
                                             params.discriminator_X)
        Y_wass_penalty = wasserstein_penalty(discrim_Y, Y_dist, Y_hat, params,
                                             params.discriminator_Y)
        tf.summary.scalar("X_wass_penalty", X_wass_penalty)
        tf.summary.scalar("Y_wass_penalty", Y_wass_penalty)

    if params.lp_distance == "l0.5":
      X_reconstr_err = (X_dist - X_reconstruction)**0.5
      Y_reconstr_err = (Y_dist - Y_reconstruction)**0.5
    elif params.lp_distance == "l2":
      X_reconstr_err = (X_dist - X_reconstruction)**2
      Y_reconstr_err = (Y_dist - Y_reconstruction)**2
    elif params.lp_distance == "l1":
      X_reconstr_err = tf.abs(X_dist - X_reconstruction)
      Y_reconstr_err = tf.abs(Y_dist - Y_reconstruction)

    X_mask_counts = tf.reduce_sum(tf.reduce_prod(X_mask, axis=2), axis=1)
    Y_mask_counts = tf.reduce_sum(tf.reduce_prod(Y_mask, axis=2), axis=1)

    X_reconstr_err = tf.reduce_sum(X_reconstr_err, axis=2)
    Y_reconstr_err = tf.reduce_sum(Y_reconstr_err, axis=2)

    X_reconstr_err = tf.reduce_mean(X_reconstr_err)
    Y_reconstr_err = tf.reduce_mean(Y_reconstr_err)

    cycle_loss = X_reconstr_err + Y_reconstr_err

    generator_loss_X = X_hat_discrim_loss + params.cycle_loss * cycle_loss
    generator_loss_Y = Y_hat_discrim_loss + params.cycle_loss * cycle_loss
    discriminator_loss_X = (X_true_discrim_loss + X_past_discrim_loss) / 2
    discriminator_loss_Y = (Y_true_discrim_loss + Y_past_discrim_loss) / 2
    total_loss = (generator_loss_X + generator_loss_Y + discriminator_loss_X +
                  discriminator_loss_Y)

    if params.use_wasserstein:
      discriminator_loss_X += X_wass_penalty
      discriminator_loss_Y += Y_wass_penalty

    if params.use_embeddings:
      embedding_loss = discriminator_loss_X + discriminator_loss_Y + params.cycle_loss * cycle_loss
      tf.summary.scalar("embedding_loss", embedding_loss)

    tf.summary.scalar("X_reconstr_err", X_reconstr_err)
    tf.summary.scalar("Y_reconstr_err", Y_reconstr_err)
    tf.summary.scalar("generator_loss_X", generator_loss_X)
    tf.summary.scalar("generator_loss_Y", generator_loss_Y)
    tf.summary.scalar("discriminator_loss_X", discriminator_loss_X)
    tf.summary.scalar("discriminator_loss_Y", discriminator_loss_Y)
    tf.summary.scalar("lr", lr)
    tf.summary.scalar("cycle_loss", params.cycle_loss)

    gs = tf.contrib.framework.get_global_step()
    if params.optimizer == "adagrad":
      optimizer = tf.train.AdagradOptimizer(lr)
    elif params.optimizer == "adam":
      optimizer = tf.train.AdamOptimizer(
          lr, beta1=params.beta1, beta2=params.beta2)
    elif params.optimizer == "rsmprop":
      optimizer = tf.train.RMSPropOptimizer(lr, momentum=params.momentum)
    elif params.optimizer == "adadelta":
      optimizer = tf.train.AdadeltaOptimizer(lr)
    elif params.optimizer == "mom":
      optimizer = tf.train.MomentumOptimizer(lr, momentum=params.momentum)

    aux_weights = [get_embedding_var(
        params, reuse=True)] if params.use_embeddings else []

    for weight in tf.trainable_variables():
      print(weight.name)

    if params.use_embeddings:
      train_emb = optimizer.minimize(embedding_loss, var_list=aux_weights)
    else:
      train_emb = tf.no_op()
    train_gX = optimizer.minimize(
        generator_loss_X, var_list=generator_G_weights)
    train_gY = optimizer.minimize(
        generator_loss_Y, var_list=generator_F_weights)
    train_dX = optimizer.minimize(
        discriminator_loss_X, var_list=discrim_X_weights)
    train_dY = optimizer.minimize(
        discriminator_loss_Y, var_list=discrim_Y_weights)

    if params.d_step == 1:
      train_op = tf.group(train_emb, train_gX, train_gY, train_dX, train_dY,
                          tf.assign_add(gs, 1))
    else:
      train_op = tf.cond(
          tf.equal(tf.mod(gs, (params.d_step + 1)), 1),
          lambda: tf.group(train_gX, train_gY, tf.assign_add(gs, 1)),
          lambda: tf.group(train_dX, train_dY, tf.assign_add(gs, 1)))

    return X_hat, Y_hat, total_loss, train_op

  def cycle_gan(features, _, mode):
    """The basic CycleGAN template.

    Args:
      features: a dict containing key "X" and key "Y"
      mode: training, evaluation or infer
    """
    with tf.variable_scope("cycle_gan"):
      # gather transformations and descriminators
      F = _TRANSFORMATIONS[params.F.name]
      G = _TRANSFORMATIONS[params.G.name]
      discrim_X = _DISCRIMINATORS[params.discriminator_X.name]
      discrim_Y = _DISCRIMINATORS[params.discriminator_Y.name]

      X = features["X"]
      Y = features["Y"]

      past_Xs = tf.placeholder(tf.float32, [None] + params.input_shape)
      past_Ys = tf.placeholder(tf.float32, [None] + params.input_shape)

      if params.type == "text":
        X_ground_truth = features["X_ground_truth"]
        Y_ground_truth = features["Y_ground_truth"]
        X_hat, Y_hat, total_loss, train_op = cycle_gan_text(
            F, G, discrim_X, discrim_Y, X, Y, X_ground_truth, Y_ground_truth,
            past_Xs, past_Ys, mode)
      else:
        raise Exception("Unsupported data type %s for CycleGAN." % params.type)

      cyclegan_hook = CycleGANHook({
          "X": X_hat,
          "Y": Y_hat
      }, {"X": past_Xs,
          "Y": past_Ys}, params)

      return tf.contrib.learn.ModelFnOps(
          mode=mode,
          predictions={"X": X_hat,
                       "Y": Y_hat},
          loss=total_loss,
          train_op=train_op,
          training_hooks=[cyclegan_hook])

  return cycle_gan
