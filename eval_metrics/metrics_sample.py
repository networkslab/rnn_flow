import numpy as np
from eval_metrics.utils import *
from os.path import join


def eval_samples(args, horizons=None):
    dataset = args.dataset
    algorithm = args.rnn_type

    sample_based = False

    result = np.load(args.output_filename)
    # print(result.files)

    true = result['groundtruth']
    pred_samples = result['predictions_samples']

    # print(true.shape)
    # print(pred_samples.shape)

    mu = np.mean(pred_samples, axis=3)
    sigma = np.std(pred_samples, axis=3)
    try:
        test_mask = np.load(join(args.data_dir, args.dataset, "test_mask_y.npz"))["test_mask"]
        test_mask = np.transpose(test_mask, [1, 0, 2])
    except Exception:
        test_mask = np.ones_like(true)
    # print(test_mask.shape)

    print('------------------------------------------------------------------------------------')
    print('Dataset : ' + dataset)
    print('Algorithm : ' + algorithm)
    
    print('------------------------------------------------------------------------------------')
    print('Horizon : 15/30/45/60 minutes')

    if horizons is None:
        if pred_samples.shape[0] == 12:
            horizons = [2, 5, 8, 11] 
        elif pred_samples.shape[0] == 4:
            horizons = [0, 1, 2, 3]
        else:
            raise NotImplementedError("Plase specify the horizons to evaluate")
    
    metrics = np.zeros((3, 4))
    results = ["Metrics & CRPS & P10QL & P90QL"]

    for horizon in horizons:
        if sample_based:
            crps = masked_crps_samples(pred_samples[horizon, :, :, :], true[horizon, :, :], test_mask[horizon, :, :])
            #coverage = empirical_coverage_samples(pred_samples[horizon, :, :, :], true[horizon, :, :], test_mask[horizon, :, :], interval=0.95)
            loss_10 = percentage_quantile_loss_samples(pred_samples[horizon, :, :, :], true[horizon, :, :], test_mask[horizon, :, :], q=0.1)
            loss_90 = percentage_quantile_loss_samples(pred_samples[horizon, :, :, :], true[horizon, :, :], test_mask[horizon, :, :], q=0.9)
        else:
            crps = masked_crps_dist(mu[horizon, :, :], sigma[horizon, :, :], true[horizon, :, :], test_mask[horizon, :, :])
            #coverage = empirical_coverage_dist(mu[horizon, :, :], sigma[horizon, :, :], true[horizon, :, :], test_mask[horizon, :, :], interval=0.95)
            loss_10 = percentage_quantile_loss_dist(mu[horizon, :, :], sigma[horizon, :, :], true[horizon, :, :], test_mask[horizon, :, :], q=0.1)
            loss_90 = percentage_quantile_loss_dist(mu[horizon, :, :], sigma[horizon, :, :], true[horizon, :, :], test_mask[horizon, :, :], q=0.9)

        
        metrics[0, int((horizon - 2) / 3)] = crps
        metrics[1, int((horizon - 2) / 3)] = loss_10
        metrics[2, int((horizon - 2) / 3)] = loss_90

        print('CRPS, P10QL, P90QL : &%.2f &%.2f &%.2f' % (crps, loss_10, loss_90))
        results.append("Horizon-{}".format(horizon) + '& %.2f   & %.2f   & %.2f' % (crps, loss_10, loss_90))
        #print('CRPS, Coverage(95), P10QL, P90QL : &%.2f &%.2f &%.2f &%.2f' % (crps, coverage, loss_10, loss_90))
        #print('P10QL, P90QL : &%.2f &%.2f' % (loss_10, loss_90))


    np.set_printoptions(precision=2)
    # print(metrics)
    return results

    












if __name__ == "__main__":

    dataset = 'PEMS03'
    algorithm = 'pgagru_nll'
    sample_based = False

    result = np.load('/home/soumyasundar/CRPS/log/' + algorithm + '_predictions_' + dataset + '.npz')
    print(result.files)

    true = result['groundtruth']
    pred_samples = result['predictions_samples']

    print(true.shape)
    print(pred_samples.shape)

    mu = np.mean(pred_samples, axis=3)
    sigma = np.std(pred_samples, axis=3)
    test_mask = np.load('/home/soumyasundar/CRPS/log/test_mask_y_' + dataset + '.npz')['test_mask']
    test_mask = np.transpose(test_mask, [1, 0, 2])
    print(test_mask.shape)

    print('------------------------------------------------------------------------------------')
    print('Dataset : ' + dataset)
    print('Algorithm : ' + algorithm)
    
    print('------------------------------------------------------------------------------------')
    print('Horizon : 15/30/45/60 minutes')
    
    metrics = np.zeros((3, 4))

    for horizon in [2, 5, 8, 11]:
        if sample_based:
            crps = masked_crps_samples(pred_samples[horizon, :, :, :], true[horizon, :, :], test_mask[horizon, :, :])
            #coverage = empirical_coverage_samples(pred_samples[horizon, :, :, :], true[horizon, :, :], test_mask[horizon, :, :], interval=0.95)
            loss_10 = percentage_quantile_loss_samples(pred_samples[horizon, :, :, :], true[horizon, :, :], test_mask[horizon, :, :], q=0.1)
            loss_90 = percentage_quantile_loss_samples(pred_samples[horizon, :, :, :], true[horizon, :, :], test_mask[horizon, :, :], q=0.9)
        else:
            crps = masked_crps_dist(mu[horizon, :, :], sigma[horizon, :, :], true[horizon, :, :], test_mask[horizon, :, :])
            #coverage = empirical_coverage_dist(mu[horizon, :, :], sigma[horizon, :, :], true[horizon, :, :], test_mask[horizon, :, :], interval=0.95)
            loss_10 = percentage_quantile_loss_dist(mu[horizon, :, :], sigma[horizon, :, :], true[horizon, :, :], test_mask[horizon, :, :], q=0.1)
            loss_90 = percentage_quantile_loss_dist(mu[horizon, :, :], sigma[horizon, :, :], true[horizon, :, :], test_mask[horizon, :, :], q=0.9)

        
        metrics[0, int((horizon - 2) / 3)] = crps
        metrics[1, int((horizon - 2) / 3)] = loss_10
        metrics[2, int((horizon - 2) / 3)] = loss_90

        print('CRPS, P10QL, P90QL : &%.2f &%.2f &%.2f' % (crps, loss_10, loss_90))
        #print('CRPS, Coverage(95), P10QL, P90QL : &%.2f &%.2f &%.2f &%.2f' % (crps, coverage, loss_10, loss_90))
        #print('P10QL, P90QL : &%.2f &%.2f' % (loss_10, loss_90))


    np.set_printoptions(precision=2)
    print(metrics)
