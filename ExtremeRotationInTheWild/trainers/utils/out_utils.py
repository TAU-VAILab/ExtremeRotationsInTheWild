import numpy as np

def statistics_from_res(res_error):
    all_res = {}
    for k, v in res_error.items():
                if v.size == 0:
                    continue
                if np.ma.is_masked(v) and v.mask.all():
                    mean, median, error_max, std, percent_10, per_from_all = 0, 0, 0, 0, 0, 0
                else:
                    mean = np.ma.mean(v)
                    median = np.ma.median(v)
                    error_max = np.ma.max(v)
                    std = np.ma.std(v)
                    count_5 = (v<5).sum(axis=0) if (k=='rotation_geodesic_error' or k=='gt_angle') else (v.compressed()<5).sum(axis=0)
                    count_10 = (v<10).sum(axis=0) if (k=='rotation_geodesic_error' or k=='gt_angle') else (v.compressed()<10).sum(axis=0)
                    count_15 = (v<15).sum(axis=0) if (k=='rotation_geodesic_error' or k=='gt_angle') else (v.compressed()<15).sum(axis=0)
                    count_30 = (v<30).sum(axis=0) if (k=='rotation_geodesic_error' or k=='gt_angle') else (v.compressed()<30).sum(axis=0)
                    percent_5 = np.true_divide(count_5, v.shape[0]) if (k=='rotation_geodesic_error' or k=='gt_angle') else np.true_divide(count_5, v.compressed().shape[0])
                    percent_10 = np.true_divide(count_10, v.shape[0]) if (k=='rotation_geodesic_error' or k=='gt_angle') else np.true_divide(count_10, v.compressed().shape[0])
                    percent_15 = np.true_divide(count_15, v.shape[0]) if (k=='rotation_geodesic_error' or k=='gt_angle') else np.true_divide(count_15, v.compressed().shape[0])
                    percent_30 = np.true_divide(count_30, v.shape[0]) if (k=='rotation_geodesic_error' or k=='gt_angle') else np.true_divide(count_30, v.compressed().shape[0])
                    per_from_all = np.true_divide(v.shape[0], res_error["gt_angle"].shape[0]) if (k=='rotation_geodesic_error' or k=='gt_angle') else np.true_divide(v.compressed().shape[0], res_error["gt_angle"].shape[0])
                all_res.update({k + '/mean': mean, k + '/median': median, k + '/max': error_max, k + '/std': std,
                                k +'/5deg': percent_5,k + '/10deg': percent_10,k +'/15deg': percent_15,k +'/30deg': percent_30,k+ '/per_from_all': per_from_all})
    return all_res