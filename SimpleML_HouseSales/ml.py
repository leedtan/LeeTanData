def apply_clipped_optimizer(opt_fcn,
                            loss,
                            clip_norm=.1,
                            clip_single=.03,
                            clip_global_norm=False):
    gvs = opt_fcn.compute_gradients(loss)

    if clip_global_norm:
        gs, vs = zip(*[(g, v) for g, v in gvs if g is not None])
        capped_gs, grad_norm_total = tf.clip_by_global_norm([g for g in gs],
                                                            clip_norm)
        capped_gvs = list(zip(capped_gs, vs))
    else:
        grad_norm_total = tf.sqrt(
            tf.reduce_sum([
                tf.reduce_sum(tf.square(grad)) for grad, var in gvs
                if grad is not None
            ]))
        capped_gvs = [(tf.clip_by_value(grad, -1 * clip_single, clip_single),
                       var) for grad, var in gvs if grad is not None]
        capped_gvs = [(tf.clip_by_norm(grad, clip_norm), var)
                      for grad, var in capped_gvs if grad is not None]

    optimizer = opt_fcn.apply_gradients(
        capped_gvs, global_step=tf.train.get_global_step())

    return optimizer, grad_norm_total
