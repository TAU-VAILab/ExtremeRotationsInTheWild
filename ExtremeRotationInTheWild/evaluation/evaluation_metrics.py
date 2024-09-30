from trainers.utils.compute_utils import *


def evaluation_metric_rotation(predict_rotation, gt_rotation,overlap_amount_array=None):
    geodesic_loss = compute_geodesic_distance_from_two_matrices(predict_rotation.view(-1, 3, 3),
                                                                gt_rotation.view(-1, 3, 3)) / pi * 180
    gt_distance = compute_angle_from_r_matrices(gt_rotation.view(-1, 3, 3))
    gt_distance = gt_distance.cpu().numpy()
    if overlap_amount_array is not None:

        geodesic_loss_overlap_none = np.ma.masked_array(geodesic_loss.cpu().numpy(), np.logical_not(np.asarray(overlap_amount_array) == "none"))
        geodesic_loss_overlap_large = np.ma.masked_array(geodesic_loss.cpu().numpy(), np.logical_not(np.asarray(overlap_amount_array) == "large"))
        geodesic_loss_overlap_small = np.ma.masked_array(geodesic_loss.cpu().numpy(), np.logical_not(np.asarray(overlap_amount_array) == "small"))
        geodesic_loss = geodesic_loss.cpu().numpy()
    else:
        geodesic_loss_overlap_large = np.ma.masked_array(geodesic_loss.cpu().numpy(), np.logical_not((gt_distance < pi / 4)))
        geodesic_loss_overlap_none = np.ma.masked_array(geodesic_loss.cpu().numpy(), np.logical_not(gt_distance > (pi / 2)))
        geodesic_loss_overlap_small = np.ma.masked_array(geodesic_loss.cpu().numpy(), np.logical_not((gt_distance >= pi / 4) & (gt_distance < pi / 2)))
        geodesic_loss = geodesic_loss.cpu().numpy()
    res_error = {
        "gt_angle": gt_distance / pi * 180,
        "rotation_geodesic_error_large": geodesic_loss_overlap_large,
        "rotation_geodesic_error_small": geodesic_loss_overlap_small,
        "rotation_geodesic_error_none": geodesic_loss_overlap_none,
        "rotation_geodesic_error": geodesic_loss,
    }
    return res_error

def evaluation_metric_rotation_euler(predict_rotation, gt_rotation,overlap_amount_array=None):
    euler_gt = compute_euler_angles_from_rotation_matrices(gt_rotation.view(-1, 3, 3))
    euler_est = compute_euler_angles_from_rotation_matrices(predict_rotation.view(-1, 3, 3))
    def angle_range(angle):
        angle[angle>=pi] -= 2*pi
        angle[angle<-pi] += 2*pi
        return angle
    euler_diff = tuple(torch.abs(angle_range(t1 - t2))/ pi * 180 for t1, t2 in zip(euler_gt, euler_est))
    res_error = {}
    for i, angle in enumerate(["yaw","pitch","roll"]):
        if overlap_amount_array is not None:
            euler_diff_none = np.ma.masked_array(euler_diff[i].cpu().numpy(), np.logical_not(np.asarray(overlap_amount_array) == "none"))
            euler_diff_overlap_large = np.ma.masked_array(euler_diff[i].cpu().numpy(), np.logical_not(np.asarray(overlap_amount_array) == "large"))
            euler_diff_overlap_small = np.ma.masked_array(euler_diff[i].cpu().numpy(), np.logical_not(np.asarray(overlap_amount_array) == "small"))
            euler_diff_angle = euler_diff[i].cpu().numpy()
        else:
            raise ValueError("overlap_amount_array is None")
        res_error.update({
            ("gt_angle_"+angle): euler_gt[i].cpu().numpy() / pi * 180,
            ("euler_diff_large_"+angle): euler_diff_overlap_large,
            ("euler_diff_small_"+angle): euler_diff_overlap_small,
            ("euler_diff_none_"+angle): euler_diff_none,
            ("euler_diff_"+angle): euler_diff_angle,
        })
    return res_error





def evaluation_metric_rotation_angle(predict_rotation, gt_rotation, gt_rmat1_array, out_rmat1_array):
    batch = predict_rotation.size(0)

    gt_yaw1, gt_pitch1 = compute_viewpoint_from_rotation_matrix(gt_rmat1_array, batch)
    gt_rmat2_array = compute_rotation_matrix_from_two_matrices(gt_rotation, gt_rmat1_array.transpose(1,2))
    gt_yaw2, gt_pitch2 = compute_viewpoint_from_rotation_matrix(gt_rmat2_array, batch)
    gt_yaw = gt_yaw2 - gt_yaw1
    gt_pitch = gt_pitch2 - gt_pitch1

    if out_rmat1_array is None:
        predict_yaw1, predict_pitch1 = compute_viewpoint_from_rotation_matrix(gt_rmat1_array, batch)
        predict_rmat2_array = compute_rotation_matrix_from_two_matrices(predict_rotation, gt_rmat1_array.transpose(1,2))    
    else: 
        predict_yaw1, predict_pitch1 = compute_viewpoint_from_rotation_matrix(out_rmat1_array, batch)
        predict_rmat2_array = compute_rotation_matrix_from_two_matrices(predict_rotation, out_rmat1_array.transpose(1,2))
    predict_yaw2, predict_pitch2 = compute_viewpoint_from_rotation_matrix(predict_rmat2_array, batch)
    predict_yaw = predict_yaw2 - predict_yaw1
    predict_pitch = predict_pitch2 - predict_pitch1
    
    def angle_range(angle):
        while (angle[angle>=pi].size(0)!=0) or (angle[angle<-pi].size(0)!=0):
            angle[angle>=pi] -= 2*pi
            angle[angle<-pi] += 2*pi
        return angle
    yaw_error = torch.abs(angle_range(gt_yaw - predict_yaw))/ pi * 180
    pitch_error = torch.abs(angle_range(gt_pitch - predict_pitch))/ pi * 180

    gt_distance = compute_angle_from_r_matrices(gt_rotation.view(-1, 3, 3))

    yaw_error_overlap_none = yaw_error[gt_distance.view(-1) > (pi / 2)]
    yaw_error_overlap_large = yaw_error[gt_distance.view(-1) < (pi / 4)]
    yaw_error_overlap_small = yaw_error[(gt_distance.view(-1) >= pi / 4) & (gt_distance.view(-1) < pi / 2)]

    pitch_error_overlap_none = pitch_error[gt_distance.view(-1) > (pi / 2)]
    pitch_error_overlap_large = pitch_error[gt_distance.view(-1) < (pi / 4)]
    pitch_error_overlap_small = pitch_error[(gt_distance.view(-1) >= pi / 4) & (gt_distance.view(-1) < pi / 2)]

    res_error = {
        "rotation_yaw_error_large": yaw_error_overlap_large,
        "rotation_yaw_error_small": yaw_error_overlap_small,
        "rotation_yaw_error_none": yaw_error_overlap_none,
        "rotation_yaw_error": yaw_error,
        "rotation_pitch_error_large": pitch_error_overlap_large,
        "rotation_pitch_error_small": pitch_error_overlap_small,
        "rotation_pitch_error_none": pitch_error_overlap_none,
        "rotation_pitch_error": pitch_error
    }
    return res_error
