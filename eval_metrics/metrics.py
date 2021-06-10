import numpy as np
from eval_metrics.utils import *
from os.path import join



def eval_deter(args, horizons=None):

    dataset = args.dataset
    algorithm = args.rnn_type

    result = np.load(args.output_filename, allow_pickle=True)

    true = result['groundtruth']
    pred = result['predictions']

    # print(true.shape)
    # print(pred.shape)
    # print(result['predictions_samples'].shape)

    try:
        test_mask = np.load(join(args.data_dir, args.dataset, "test_mask_y.npz"))["test_mask"]
        test_mask = np.transpose(test_mask, [1, 0, 2])
    except Exception:
        test_mask = np.ones_like(true)

    print(test_mask.shape)

    print('------------------------------------------------------------------------------------')
    print('Dataset : ' + dataset)
    print('Algorithm : ' + algorithm)
    print('------------------------------------------------------------------------------------')
    print('Horizon : 15/30/45/60 minutes')

    metrics = np.zeros((3, 4))
    results = ["Metrics & MAE & MAPE & RMSE"]

    if horizons is None:
        if pred.shape[0] == 12:
            horizons = [2, 5, 8, 11] 
        elif pred.shape[0] == 4:
            horizons = [0, 1, 2, 3]
        else:
            raise NotImplementedError("Plase specify the horizons to evaluate")

    for horizon in horizons:

        mae = masked_mae_np(preds=pred[horizon, :, :], labels=true[horizon, :, :], mask=test_mask[horizon, :, :])
        mape = masked_mape_np(preds=pred[horizon, :, :], labels=true[horizon, :, :], mask=test_mask[horizon, :, :])
        rmse = masked_rmse_np(preds=pred[horizon, :, :], labels=true[horizon, :, :], mask=test_mask[horizon, :, :])
        
        metrics[0, int((horizon - 2) / 3)] = mae
        metrics[1, int((horizon - 2) / 3)] = mape
        metrics[2, int((horizon - 2) / 3)] = rmse

        print('MAE, MAPE, RMSE : &%.2f &%.2f &%.2f' % (mae, mape, rmse))
        results.append("Horizon-{}".format(horizon)+ ' & %.2f  & %.2f  & %.2f' % (mae, mape, rmse))

    np.set_printoptions(precision=2)     

    return results
    # print(metrics)













if __name__ == "__main__":

    dataset = 'PEMS08'
    algorithm = 'pgagru_mae'

    result = np.load('/home/soumyasundar/CRPS/log/' + algorithm + '_predictions_' + dataset + '.npz', allow_pickle=True)
    print(result.files)

    # res = result['arr_0']
    # result = res.tolist()
    # print(result.keys)

    true = result['groundtruth']
    pred = result['predictions']

    # true = np.transpose(np.tile(result['truth'][:, :, np.newaxis], [1, 1, 12]), [2, 0, 1])
    # # pred = np.transpose(np.tile(result['pred'][:, :, np.newaxis], [1, 1, 12]), [2, 0, 1])
    # pred = np.transpose(np.repeat(result['pred'], 3, axis=2), [2, 0, 1])

    # true = np.transpose(np.squeeze(result['ground_truth']),  [2, 0, 1])
    # pred = np.transpose(np.squeeze(result['prediction']), [2, 0, 1])

    # true = np.transpose(result['truth'], [1, 0, 2])
    # pred = np.transpose(result['preds'], [1, 0, 2])

    # true = np.transpose(np.repeat(np.squeeze(result['ground_truth']), 3, axis=1), [1, 0, 2])
    # pred = np.transpose(np.repeat(np.squeeze(result['predictions']), 3, axis=0), [0, 1, 2])

    print(true.shape)
    print(pred.shape)
    print(result['predictions_samples'].shape)

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

        mae = masked_mae_np(preds=pred[horizon, :, :], labels=true[horizon, :, :], mask=test_mask[horizon, :, :])
        mape = masked_mape_np(preds=pred[horizon, :, :], labels=true[horizon, :, :], mask=test_mask[horizon, :, :])
        rmse = masked_rmse_np(preds=pred[horizon, :, :], labels=true[horizon, :, :], mask=test_mask[horizon, :, :])
        
        metrics[0, int((horizon - 2) / 3)] = mae
        metrics[1, int((horizon - 2) / 3)] = mape
        metrics[2, int((horizon - 2) / 3)] = rmse

        print('MAE, MAPE, RMSE : &%.2f &%.2f &%.2f' % (mae, mape, rmse))

    np.set_printoptions(precision=2)     
    print(metrics)