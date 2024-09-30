import torch
import torch.nn.functional as F
from math import pi
import numpy as np
from pytorch3d.transforms import euler_angles_to_matrix,matrix_to_euler_angles
from scipy.spatial.transform import Rotation
from scipy.ndimage import shift, gaussian_filter
from scipy.signal import argrelextrema


def qvec2rotmat(qvec,batch):
    r11 = (1 - 2 * qvec[2]**2 - 2 * qvec[3]**2).float().view(batch, 1)
    r12 = (2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3]).float().view(batch, 1)
    r13 = (2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]).float().view(batch, 1)
    r21 = (2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3]).float().view(batch, 1)
    r22 = (1 - 2 * qvec[1]**2 - 2 * qvec[3]**2).view(batch, 1)
    r23 = (2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]).float().view(batch, 1)
    r31 = (2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2]).float().view(batch, 1)
    r32 = (2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1]).float().view(batch, 1)
    r33 = (1 - 2 * qvec[1]**2 - 2 * qvec[2]**2).float().view(batch, 1)
    row1 = torch.cat((r11, r12, r13), 1).view(-1, 1, 3)
    row2 = torch.cat((r21, r22, r23), 1).view(-1, 1, 3)
    row3 = torch.cat((r31, r32, r33), 1).view(-1, 1, 3)
    matrix = torch.cat((row1,row2,row3), 1).type(torch.FloatTensor)  # batch*3*3
    trans_mat = np.array([[ 0,  -1, 0], \
                         [ 1, 0,  0], \
                         [0, 0,1]])
    matrix = torch.matmul(torch.matmul(torch.from_numpy(trans_mat).type(torch.float),matrix),torch.from_numpy(trans_mat.T).type(torch.float))
    return matrix

def compute_gt_rmat_colmap(q1, q2, batch_size):
    gt_mtx1= qvec2rotmat(q1,batch_size)
    gt_mtx2= qvec2rotmat(q2,batch_size)
    gt_rmat_matrix = compute_rotation_matrix_from_two_matrices(gt_mtx2, gt_mtx1).view(batch_size, 3, 3)
    return gt_rmat_matrix.cuda()

def compute_gt_rmat(rotation_x1, rotation_y1, rotation_x2, rotation_y2, batch_size):
    gt_mtx1 = compute_rotation_matrix_from_viewpoint(rotation_x1, rotation_y1, batch_size).view(batch_size, 3, 3)
    gt_mtx2 = compute_rotation_matrix_from_viewpoint(rotation_x2, rotation_y2, batch_size).view(batch_size, 3, 3)
    gt_rmat_matrix = compute_rotation_matrix_from_two_matrices(gt_mtx2, gt_mtx1).view(batch_size, 3, 3)
    return gt_rmat_matrix.cuda()

def compute_angle(rotation_x1, rotation_x2, rotation_y1, rotation_y2):
    delta_x = rotation_x2 - rotation_x1
    delta_x[delta_x >= pi] -= 2 * pi
    delta_x[delta_x < -pi] += 2 * pi
    delta_y = rotation_y1
    delta_y[delta_y >= pi] -= 2 * pi
    delta_y[delta_y < -pi] += 2 * pi
    delta_z = rotation_y2
    delta_z[delta_z >= pi] -= 2 * pi
    delta_z[delta_z < -pi] += 2 * pi
    return delta_x, delta_y, delta_z

def compute_yaw_angle(rotation_x1, rotation_x2):
    delta_x = rotation_x2 - rotation_x1
    delta_x[delta_x >= pi] -= 2 * pi
    delta_x[delta_x < -pi] += 2 * pi
    return delta_x
    

def compute_out_rmat(out_rotation_x, out_rotation_y, out_rotation_z, batch_size):
    if out_rotation_x.size(-1) == 1:
        rt1 = compute_rotation_matrix_from_viewpoint(torch.zeros(out_rotation_x.size()).to(out_rotation_x),
                                                     out_rotation_y.float(),
                                                     batch_size).view(batch_size, 3, 3)
        rt2 = compute_rotation_matrix_from_viewpoint(out_rotation_x.float(),
                                                     out_rotation_z.float(),
                                                     batch_size).view(batch_size, 3, 3)
    else:
        _, rotation_x = torch.topk(out_rotation_x, 1, dim=-1)
        _, rotation_y = torch.topk(out_rotation_y, 1, dim=-1)
        _, rotation_z = torch.topk(out_rotation_z, 1, dim=-1)
        rt1 = compute_rotation_matrix_from_viewpoint(torch.zeros(rotation_x.size()).to(rotation_x),
                                                     rotation_y.float() / out_rotation_y.size(-1) * 2 * pi - pi,
                                                     batch_size).view(batch_size, 3, 3)
        rt2 = compute_rotation_matrix_from_viewpoint(rotation_x.float() / out_rotation_x.size(-1) * 2 * pi - pi,
                                                     rotation_z.float() / out_rotation_z.size(-1) * 2 * pi - pi,
                                                     batch_size).view(batch_size, 3, 3)
    out_rmat = compute_rotation_matrix_from_two_matrices(rt2, rt1).view(batch_size, 3, 3).cuda()
    return out_rmat, rt1.cuda()



def compute_out_rmat_from_euler(out_rotation_x, out_rotation_y, out_rotation_z, batch_size):
    if out_rotation_x.size(-1) == 1:
        out_rmat = compute_rotation_matrix_from_euler_angle(out_rotation_x.float(), out_rotation_y.float(), out_rotation_z.float(), batch_size)
    else:
        _, rotation_x = torch.topk(out_rotation_x, 1, dim=-1)
        _, rotation_y = torch.topk(out_rotation_y, 1, dim=-1)
        _, rotation_z = torch.topk(out_rotation_z, 1, dim=-1)
        out_rmat = compute_rotation_matrix_from_euler_angle(rotation_x.float() / out_rotation_x.size(-1) * 2 * pi - pi,
                                                                rotation_y.float() / out_rotation_y.size(-1) * 2 * pi - pi,
                                                                rotation_z.float() / out_rotation_z.size(-1) * 2 * pi - pi, batch_size)
    return out_rmat.view(batch_size, 3, 3).cuda(), None



def return_maximas(rotation_ang_dist,top_pk_dim):
    m = torch.nn.Softmax(dim=1)
    ang_prob = m(rotation_ang_dist)
    top_k_maximas_idxs = []
    top_k_maximas_values_per_angle = []
    for i in range(ang_prob.shape[0]):
        smoothed_signal = gaussian_filter(ang_prob[i].cpu().numpy(), sigma=5)
        local_maxima_indices = argrelextrema(smoothed_signal, np.greater)
        border_indices = [0, len(smoothed_signal) - 1]
        border_indices = [b for b in border_indices if np.sum(np.abs(local_maxima_indices[0]-b)<10) == 0]
        local_maxima_indices = np.concatenate((border_indices, local_maxima_indices[0])).astype(np.int32)
        local_maxima_values = smoothed_signal[local_maxima_indices]
        if len(local_maxima_indices) < top_pk_dim:
            top_k_maximas_idxs.append(torch.from_numpy(local_maxima_indices))
            top_k_maximas_values_per_angle.append(torch.from_numpy(local_maxima_values))
            continue
        max_indices = np.argsort(local_maxima_values)[-top_pk_dim:]
        top_k_maxima_values = local_maxima_values[max_indices] 
        top_k_maxima_indices = local_maxima_indices[max_indices] 
        top_k_maximas_idxs.append( torch.from_numpy(top_k_maxima_indices))
        top_k_maximas_values_per_angle.append(torch.from_numpy(top_k_maxima_values))
    return (top_k_maximas_idxs,top_k_maximas_values_per_angle)

def compute_out_rmat_from_euler_top_k(out_rotation_x, out_rotation_y, out_rotation_z, batch_size,gt_mat,top_picks = 3,angle_z=None):
    if out_rotation_x.size(-1) == 1:
        out_rmat = compute_rotation_matrix_from_euler_angle(out_rotation_x.float(), out_rotation_y.float(), out_rotation_z.float(), batch_size)
    else:
        out_rmat = torch.zeros_like(gt_mat)
        top_pk_dim = top_picks
        rotation_x,rotation_x_prob = return_maximas(out_rotation_x,top_pk_dim)
        _, rotation_y = torch.topk(out_rotation_y, 1, dim=-1)
        if angle_z is not None:
            rotation_z = angle_z.unsqueeze(1)
        else:
            _, rotation_z = torch.topk(out_rotation_z, 1, dim=-1)
            
        for i in range(out_rmat.shape[0]):
            top_pk_dim = rotation_x[i].shape[0]
            if angle_z is not None:
                out_rmat_ro_img = compute_rotation_matrix_from_euler_angle(rotation_x[i].float() / out_rotation_x.size(-1) * 2 * pi - pi,
                                                                rotation_y[i].expand(top_pk_dim,-1).float() / out_rotation_y.size(-1) * 2 * pi - pi,
                                                                rotation_z[i].expand(top_pk_dim,-1).float() , top_pk_dim)
            else:
                out_rmat_ro_img = compute_rotation_matrix_from_euler_angle(rotation_x[i].float() / out_rotation_x.size(-1) * 2 * pi - pi,
                                                                rotation_y[i].expand(top_pk_dim,-1).float() / out_rotation_y.size(-1) * 2 * pi - pi,
                                                                rotation_z[i].expand(top_pk_dim,-1).float() / out_rotation_z.size(-1) * 2 * pi - pi, top_pk_dim)
            geodesic_loss = compute_geodesic_distance_from_two_matrices(out_rmat_ro_img.view(-1, 3, 3).cuda(),
                                                                gt_mat[i].expand(top_pk_dim,-1,-1).view(-1, 3, 3)) / pi * 180
            index_win = torch.argmin(geodesic_loss)
            out_rmat[i] = out_rmat_ro_img[index_win]
    return out_rmat.view(batch_size, 3, 3).cuda(), rotation_x,rotation_x_prob



def compute_rotation_matrix_from_viewpoint(rotation_x, rotation_y, batch):
    rotax = rotation_x.view(batch, 1).type(torch.FloatTensor)
    rotay = - rotation_y.view(batch, 1).type(torch.FloatTensor)
    # rotaz = torch.zeros(batch, 1)

    c1 = torch.cos(rotax).view(batch, 1)  # batch*1
    s1 = torch.sin(rotax).view(batch, 1)  # batch*1
    c2 = torch.cos(rotay).view(batch, 1)  # batch*1
    s2 = torch.sin(rotay).view(batch, 1)  # batch*1

    # pitch --> yaw
    row1 = torch.cat((c2, s1 * s2, c1 * s2), 1).view(-1, 1, 3)  # batch*1*3
    row2 = torch.cat((torch.autograd.Variable(torch.zeros(s2.size())), c1, -s1), 1).view(-1, 1, 3)  # batch*1*3
    row3 = torch.cat((-s2, s1 * c2, c1 * c2), 1).view(-1, 1, 3)  # batch*1*3

    matrix = torch.cat((row1, row2, row3), 1)  # batch*3*3

    return matrix




def compute_viewpoint_from_rotation_matrix(rotation_matrix, batch):
    # pitch --> yaw
    s1 = - rotation_matrix.view(batch, 3, 3)[:, 1, 2]
    c1 = rotation_matrix.view(batch, 3, 3)[:, 1, 1]
    s2 = - rotation_matrix.view(batch, 3, 3)[:, 2, 0]
    c2 = rotation_matrix.view(batch, 3, 3)[:, 0, 0]
    # rotation_x = torch.asin(s1).view(batch, 1)
    rotation_x = torch.acos(c1).view(batch, 1)
    index = torch.nonzero(s1.view(-1) < 0, as_tuple=True)
    rotation_x[index] = -rotation_x[index]
    rotation_y = torch.acos(c2).view(batch, 1)
    indexy = torch.nonzero(s2.view(-1) < 0, as_tuple=True)
    rotation_y[indexy] = -rotation_y[indexy]
    rotation_y = - rotation_y
    return rotation_x, rotation_y



def compute_rotation_matrix_from_two_matrices(m1, m2):
    batch = m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3
    return m


def compute_correlation_volume_pairwise(fmap1, fmap2, num_levels):
    batch, dim, ht, wd = fmap1.shape
    fmap1 = fmap1.view(batch, dim, ht * wd)
    fmap2 = fmap2.view(batch, dim, ht * wd)

    corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
    corr = corr.view(batch, ht, wd, 1, ht, wd)
    corr = corr / torch.sqrt(torch.tensor(dim).float())

    batch2, h1, w1, dim2, h2, w2 = corr.shape
    corr = corr.reshape(batch2 * h1 * w1, dim2, h2, w2)
    corr_pyramid = []
    corr_pyramid.append(corr)
    for i in range(num_levels - 1):
        corr = F.avg_pool2d(corr, 2, stride=2)
        corr_pyramid.append(corr)

    out_pyramid = []
    for i in range(num_levels):
        corr = corr_pyramid[i]
        corr = corr.view(batch2, h1, w1, -1)
        out_pyramid.append(corr)
        out = torch.cat(out_pyramid, dim=-1)
    return out.permute(0, 3, 1, 2).contiguous().float()


'''Tools in RotationContinuity.
https://github.com/papagina/RotationContinuity/blob/master/shapenet/code/tools.py
Reference:
[1] Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, Hao Li
    On The Continuity of Rotation Representations in Neural Networks. arXiv:1812.07035
'''

# batch*n
def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    return v


# u, v batch*n
def cross_product(u, v):
    batch = u.shape[0]
    # print (u.shape)
    # print (v.shape)
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3

    return out


# poses batch*6
# poses
def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix


# in a batch*5, axis int
def stereographic_unproject(a, axis=None):
    """
	Inverse of stereographic projection: increases dimension by one.
	"""
    batch = a.shape[0]
    if axis is None:
        axis = a.shape[1]
    s2 = torch.pow(a, 2).sum(1)  # batch
    ans = torch.autograd.Variable(torch.zeros(batch, a.shape[1] + 1).cuda())  # batch*6
    unproj = 2 * a / (s2 + 1).view(batch, 1).repeat(1, a.shape[1])  # batch*5
    if (axis > 0):
        ans[:, :axis] = unproj[:, :axis]  # batch*(axis-0)
    ans[:, axis] = (s2 - 1) / (s2 + 1)  # batch
    ans[:, axis + 1:] = unproj[:,
                        axis:]  # batch*(5-axis)		# Note that this is a no-op if the default option (last axis) is used
    return ans


# a batch*5
# out batch*3*3
def compute_rotation_matrix_from_ortho5d(a):
    batch = a.shape[0]
    proj_scale_np = np.array([np.sqrt(2) + 1, np.sqrt(2) + 1, np.sqrt(2)])  # 3
    proj_scale = torch.autograd.Variable(torch.FloatTensor(proj_scale_np).cuda()).view(1, 3).repeat(batch, 1)  # batch,3

    u = stereographic_unproject(a[:, 2:5] * proj_scale, axis=0)  # batch*4
    norm = torch.sqrt(torch.pow(u[:, 1:], 2).sum(1))  # batch
    u = u / norm.view(batch, 1).repeat(1, u.shape[1])  # batch*4
    b = torch.cat((a[:, 0:2], u), 1)  # batch*6
    matrix = compute_rotation_matrix_from_ortho6d(b)
    return matrix


# quaternion batch*4
def compute_rotation_matrix_from_quaternion(quaternion):
    batch = quaternion.shape[0]

    quat = normalize_vector(quaternion)

    qw = quat[..., 0].view(batch, 1)
    qx = quat[..., 1].view(batch, 1)
    qy = quat[..., 2].view(batch, 1)
    qz = quat[..., 3].view(batch, 1)

    # Unit quaternion rotation matrices computatation
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    xw = qx * qw
    yw = qy * qw
    zw = qz * qw

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), 1)  # batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), 1)  # batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), 1)  # batch*3

    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(batch, 1, 3)), 1)  # batch*3*3

    return matrix


# axisAngle batch*4 angle, x,y,z
def compute_rotation_matrix_from_axisAngle(axisAngle):
    batch = axisAngle.shape[0]

    theta = torch.tanh(axisAngle[:, 0]) * np.pi  # [-180, 180]
    sin = torch.sin(theta)
    axis = normalize_vector(axisAngle[:, 1:4])  # batch*3
    qw = torch.cos(theta)
    qx = axis[:, 0] * sin
    qy = axis[:, 1] * sin
    qz = axis[:, 2] * sin

    # Unit quaternion rotation matrices computatation
    xx = (qx * qx).view(batch, 1)
    yy = (qy * qy).view(batch, 1)
    zz = (qz * qz).view(batch, 1)
    xy = (qx * qy).view(batch, 1)
    xz = (qx * qz).view(batch, 1)
    yz = (qy * qz).view(batch, 1)
    xw = (qx * qw).view(batch, 1)
    yw = (qy * qw).view(batch, 1)
    zw = (qz * qw).view(batch, 1)

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), 1)  # batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), 1)  # batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), 1)  # batch*3

    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(batch, 1, 3)), 1)  # batch*3*3

    return matrix


# euler batch*4
# output cuda batch*3*3 matrices in the rotation order of XZ'Y'' (intrinsic) or YZX (extrinsic)
def compute_rotation_matrix_from_euler(euler):
    batch = euler.shape[0]

    c1 = torch.cos(euler[:, 0]).view(batch, 1)  # batch*1
    s1 = torch.sin(euler[:, 0]).view(batch, 1)  # batch*1
    c2 = torch.cos(euler[:, 2]).view(batch, 1)  # batch*1
    s2 = torch.sin(euler[:, 2]).view(batch, 1)  # batch*1
    c3 = torch.cos(euler[:, 1]).view(batch, 1)  # batch*1
    s3 = torch.sin(euler[:, 1]).view(batch, 1)  # batch*1

    row1 = torch.cat((c2 * c3, -s2, c2 * s3), 1).view(-1, 1, 3)  # batch*1*3
    row2 = torch.cat((c1 * s2 * c3 + s1 * s3, c1 * c2, c1 * s2 * s3 - s1 * c3), 1).view(-1, 1, 3)  # batch*1*3
    row3 = torch.cat((s1 * s2 * c3 - c1 * s3, s1 * c2, s1 * s2 * s3 + c1 * c3), 1).view(-1, 1, 3)  # batch*1*3

    matrix = torch.cat((row1, row2, row3), 1)  # batch*3*3

    return matrix


# matrices batch*3*3
# both matrix are orthogonal rotation matrices
# out theta between 0 to 180 degree batch
def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch = m1.shape[0]
    m = torch.bmm(m1.to(torch.float64), m2.to(torch.float64).transpose(1, 2))  # batch*3*3

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()))
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda()) * -1)

    theta = torch.acos(cos)

    return theta


# matrices batch*3*3
# both matrix are orthogonal rotation matrices
# out theta between 0 to pi batch
def compute_angle_from_r_matrices(m):
    batch = m.shape[0]

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()))
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda()) * -1)

    theta = torch.acos(cos)

    return theta


def compute_euler_angles_from_rotation_matrices(rotation_matrices):
    batch = rotation_matrices.shape[0]
    R = rotation_matrices
    sy = torch.sqrt(R[:, 0, 0] * R[:, 0, 0] + R[:, 1, 0] * R[:, 1, 0])
    singular = sy < 1e-6
    singular = singular.float()

    x = torch.atan2(R[:, 2, 1], R[:, 2, 2])
    y = torch.atan2(-R[:, 2, 0], sy)
    z = torch.atan2(R[:, 1, 0], R[:, 0, 0])

    xs = torch.atan2(-R[:, 1, 2], R[:, 1, 1])
    ys = torch.atan2(-R[:, 2, 0], sy)
    zs = R[:, 1, 0] * 0

    rotation_x = x * (1 - singular) + xs * singular
    rotation_y = y * (1 - singular) + ys * singular
    rotation_z = z * (1 - singular) + zs * singular

    return rotation_x, rotation_y, rotation_z


def compute_rotation_matrix_from_euler_angle(rotation_x, rotation_y, rotation_z=None, batch=None):
    rotax = rotation_x.view(batch, 1).type(torch.FloatTensor)
    rotay = rotation_y.view(batch, 1).type(torch.FloatTensor)
    if rotation_z is None:
        rotaz = torch.zeros(batch, 1)
    else:
        rotaz = rotation_z.view(batch, 1).type(torch.FloatTensor)
    c3 = torch.cos(rotax).view(batch, 1)  
    s3 = torch.sin(rotax).view(batch, 1)  
    c2 = torch.cos(rotay).view(batch, 1)  
    s2 = torch.sin(rotay).view(batch, 1)  
    c1 = torch.cos(rotaz).view(batch, 1)  
    s1 = torch.sin(rotaz).view(batch, 1) 

    row1 = torch.cat((c1 * c2, c1 * s2 * s3 - s1 * c3, c1 * s2 * c3 + s1 * s3), 1).view(-1, 1, 3)  # batch*1*3
    row2 = torch.cat((s1 * c2, s1 * s2 * s3 + c1 * c3, s1 * s2 * c3 - c1 * s3), 1).view(-1, 1, 3)  # batch*1*3
    row3 = torch.cat((-s2, c2 * s3, c2 * c3), 1).view(-1, 1, 3)  # batch*1*3

    matrix = torch.cat((row1, row2, row3), 1)  # batch*3*3

    return matrix