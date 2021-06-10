import numpy as np
import scipy.stats
from eval_metrics.crps import crps_ensemble, crps_gaussian


def masked_rmse_np(preds, labels, mask):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, mask=mask))


def masked_mse_np(preds, labels, mask):
    mask = mask.astype('float32')
    mask /= np.mean(mask)
    rmse = np.square(np.subtract(preds, labels)).astype('float32')
    rmse = np.nan_to_num(rmse * mask)
    return np.mean(rmse)


def masked_mae_np(preds, labels, mask):
    mask = mask.astype('float32')
    mask /= np.mean(mask)
    mae = np.abs(np.subtract(preds, labels)).astype('float32')
    mae = np.nan_to_num(mae * mask)
    return np.mean(mae)


def masked_mape_np(preds, labels, mask):
    mask = mask.astype('float32')
    mask /= np.mean(mask)
    # add labels masking for zeros
    zero_labels = np.where(labels == 0)
    zero_masking = np.ones_like(labels)
    zero_masking[zero_labels] = 1
    valid_num = np.sum(zero_masking)
    # 
    mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
    mape = np.nan_to_num(mask * mape)
    mape = mape * zero_masking
    return np.sum(mape) / valid_num * 100


def masked_crps_samples(pred_samples, labels, mask):
    mask = mask.astype('float32')
    mask /= np.mean(mask)

    dim1, dim2, nParticle = np.shape(pred_samples)
    pred_samples = np.reshape(pred_samples, [-1, nParticle])
    labels = np.reshape(labels, [-1])
    mask = np.reshape(mask, [-1])

    crps = np.zeros(np.shape(mask))
    for idx in range(len(mask)):
        crps[idx] = crps_ensemble(labels[idx], pred_samples[idx, :])

    crps = crps.astype('float32')
    crps = np.nan_to_num(crps * mask)
    return np.mean(crps)


def empirical_coverage_samples(pred_samples, labels, mask, interval=0.95):
    mask = mask.astype('float32')
    mask /= np.mean(mask)

    dim1, dim2, nParticle = np.shape(pred_samples)
    pred_samples = np.reshape(pred_samples, [-1, nParticle])
    labels = np.reshape(labels, [-1])
    mask = np.reshape(mask, [-1])

    lower_quantile = (1-interval)/2
    upper_quantile = 1 - (1-interval)/2

    coverage = np.zeros(np.shape(mask))
    for idx in range(len(mask)):

        lb = np.quantile(pred_samples[idx, :], lower_quantile)
        ub = np.quantile(pred_samples[idx, :], upper_quantile)

        if lb <= labels[idx] <= ub:
            coverage[idx] = 1.0

    coverage = coverage.astype('float32')
    coverage = np.nan_to_num(coverage * mask)
    return np.mean(coverage) * 100


def masked_crps_dist(mu, sigma, labels, mask):
    mask = mask.astype('float32')
    mask /= np.mean(mask)

    mu = np.reshape(mu, [-1])
    sigma = np.reshape(sigma, [-1])
    labels = np.reshape(labels, [-1])
    mask = np.reshape(mask, [-1])

    crps = np.zeros(np.shape(mask))
    for idx in range(len(mask)):
        crps[idx] = crps_gaussian(labels[idx], mu[idx], sigma[idx])

    crps = crps.astype('float32')
    crps = np.nan_to_num(crps * mask)
    return np.mean(crps)


def empirical_coverage_dist(mu, sigma, labels, mask, interval=0.95):
    mask = mask.astype('float32')
    mask /= np.mean(mask)

    mu = np.reshape(mu, [-1])
    sigma = np.reshape(sigma, [-1])
    labels = np.reshape(labels, [-1])
    mask = np.reshape(mask, [-1])

    lower_quantile = (1-interval)/2
    upper_quantile = 1 - (1-interval)/2

    lb = mu + scipy.stats.norm.ppf(lower_quantile) * sigma
    ub = mu + scipy.stats.norm.ppf(upper_quantile) * sigma

    coverage = np.zeros(np.shape(mask))
    for idx in range(len(mask)):

        if lb[idx] <= labels[idx] <= ub[idx]:
            coverage[idx] = 1.0

    coverage = coverage.astype('float32')
    coverage = np.nan_to_num(coverage * mask)
    return np.mean(coverage) * 100


def percentage_quantile_loss_samples(pred_samples, labels, mask, q=0.9):
    mask = mask.astype('float32')
    mask /= np.mean(mask)

    dim1, dim2, nParticle = np.shape(pred_samples)
    pred_samples = np.reshape(pred_samples, [-1, nParticle])
    labels = np.reshape(labels, [-1])
    mask = np.reshape(mask, [-1])

    pred_q = np.zeros(np.shape(mask))

    for idx in range(len(mask)):
        pred_q[idx] = np.quantile(pred_samples[idx, :], q)

    loss = 2 * np.maximum(q * (labels - pred_q), (q - 1) * (labels - pred_q))

    loss *= mask
    labels = np.abs(labels) * mask
    loss = loss.astype('float32')
    labels = labels.astype('float32')
    loss = np.nan_to_num(loss)
    labels = np.nan_to_num(labels)
    return np.mean(loss) / np.mean(labels) * 100


def percentage_quantile_loss_dist(mu, sigma, labels, mask, q=0.9):
    mask = mask.astype('float32')
    mask /= np.mean(mask)

    mu = np.reshape(mu, [-1])
    sigma = np.reshape(sigma, [-1])
    labels = np.reshape(labels, [-1])
    mask = np.reshape(mask, [-1])

    pred_q = mu + scipy.stats.norm.ppf(q) * sigma
    loss = 2 * np.maximum(q * (labels - pred_q), (q - 1) * (labels - pred_q))

    loss *= mask
    labels = np.abs(labels) * mask
    loss = loss.astype('float32')
    labels = labels.astype('float32')
    loss = np.nan_to_num(loss)
    labels = np.nan_to_num(labels)
    return np.mean(loss) / np.mean(labels) * 100


def percentage_quantile_loss_mqrnn(percentile, labels, mask, q=0.9):
    mask = mask.astype('float32')
    mask /= np.mean(mask)

    percentile = np.reshape(percentile, [-1])
    labels = np.reshape(labels, [-1])
    mask = np.reshape(mask, [-1])

    pred_q = percentile

    loss = 2 * np.maximum(q * (labels - pred_q), (q - 1) * (labels - pred_q))

    loss *= mask
    labels = np.abs(labels) * mask
    loss = loss.astype('float32')
    labels = labels.astype('float32')
    loss = np.nan_to_num(loss)
    labels = np.nan_to_num(labels)
    return np.mean(loss) / np.mean(labels) * 100
