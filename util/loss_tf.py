import tensorflow as tf


def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    score = tf.reduce_sum(tf.multiply(user_emb, pos_item_emb), 1) - tf.reduce_sum(tf.multiply(user_emb, neg_item_emb), 1)
    loss = -tf.reduce_sum(tf.log(tf.sigmoid(score) + 10e-8))
    return loss


def InfoNCE(view1, view2, temperature):
    pos_score = tf.reduce_sum(tf.multiply(view1, view2), axis=1)
    ttl_score = tf.matmul(view1, view2, transpose_a=False, transpose_b=True)
    pos_score = tf.exp(pos_score / temperature)
    ttl_score = tf.reduce_sum(tf.exp(ttl_score / temperature), axis=1)
    cl_loss = -tf.reduce_sum(tf.log(pos_score / ttl_score))
    return cl_loss


# Sampled Softmax
def ssm_loss(user_emb, pos_item_emb, neg_item_emb):
    user_emb = tf.nn.l2_normalize(user_emb, 1)
    pos_item_emb = tf.nn.l2_normalize(pos_item_emb, 1)
    neg_item_emb = tf.nn.l2_normalize(neg_item_emb, 1)
    pos_score = tf.reduce_sum(tf.multiply(user_emb, pos_item_emb), 1)
    ttl_score = tf.matmul(user_emb, neg_item_emb, transpose_a=False, transpose_b=True)
    ttl_score = tf.concat([tf.reshape(pos_score, (-1, 1)), ttl_score], axis=1)
    pos_score = tf.exp(pos_score / 0.2)
    ttl_score = tf.reduce_sum(tf.exp(ttl_score / 0.2), axis=1)
    return -tf.reduce_mean(tf.log(pos_score / ttl_score))
