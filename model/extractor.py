# shape_params 100
# pose_params 6
# expression_params 50
import os
import torch
from flame_pytorch import convert, draw_and_show_points, map_points

from data_process import load

def get_coords_from_params(shape: torch.Tensor,
                            pose: torch.Tensor,
                            expression: torch.Tensor):
    '''
    function that gives 2D coordinates from FLAME model based on 
    shape, pose and expression parameters

    len(pose) = 10,
    len(shape) = 100,
    len(expression) = 50,
    '''
    coords = convert(pose[:, 0:6], shape, expression, pose[:, 6:10])
    return coords


def lossRMSE(predict: torch.Tensor, actual: torch.Tensor):
    '''
    simple RMSE loss
    '''
    return torch.sqrt(torch.sum(
        (predict - actual) * (predict - actual)) / predict.shape[0]) 


def calc_grad(params, truth, init_loss, idparam, precise):
    '''
    function finds one gradient for the group from params 
    list specified by idparam
    '''
    grad = torch.zeros(params[idparam].shape[1])
    for i in range(params[idparam].shape[1]):
        oldval = params[idparam][:, i]
        params[idparam][:, i] += precise
        newloss = lossRMSE(truth, get_coords_from_params(*params))
        params[idparam][:, i] = oldval
        grad[i] = (newloss - init_loss) / precise
    # print('grad is.. ', grad)
    return grad

def calc_grads( truth: torch.Tensor, flag_shape, flag_pose, flag_expr,
                params, precise: float):
    '''
    Function calculates gradients for groups of parameters 
    that are specified by flags
    '''
    init_loss = lossRMSE(truth, get_coords_from_params(*params))
    shape_grads = torch.zeros(100)
    pose_grads = torch.zeros(10)
    expr_grads = torch.zeros(50)

    if flag_shape:
        shape_grads = calc_grad(params, truth, init_loss, 0, precise)
    if flag_pose:
        pose_grads = calc_grad(params, truth, init_loss, 1, precise)
    if flag_expr:
        expr_grads = calc_grad(params, truth, init_loss, 2, precise)

    print(shape_grads)
    return shape_grads, pose_grads, expr_grads
    

def processtruth(truth: torch.Tensor):
    '''
    function adapts truth to the format that we need (center it and scale)
    '''
    truth -= 0.5
    truth = torch.stack([torch.Tensor([th[0], -th[1]]) for th in truth])
    truth *= 2
    return truth

def extract(truth: torch.Tensor, show=False):
    '''
    Function that user gradient descent to find 
    FLAME parameters, for which FLAME face will be most similar to the truth

    '''
    truth = processtruth(truth)
    shape_p = torch.zeros(1, 100)
    pose_p = torch.zeros(1, 10)
    expr_p = torch.zeros(1, 50)

    params = (shape_p, pose_p, expr_p)
    print([p.shape for p in params])
    conv_iters_pose = 30 # count of iteration until convergence (assume)
    step_size = 0.1
    for i in range(conv_iters_pose):
        shape_grads, pose_grads, expr_grads = calc_grads(
            truth, False, True, False, params, 0.0001
        )
        pose_p -= step_size * pose_grads
        # shape_p -= step_size * shape_grads
        # expr_p -= step_size * expr_grads
        params = (shape_p, pose_p, expr_p)

        print('iter: ', i)
    conv_iters_shape = 40
    step_size = 0.2
    for i in range(conv_iters_shape):
        shape_grads, pose_grads, expr_grads = calc_grads(
            truth, True, False, False, params, 0.0001
        )
        
        pose_p -= step_size * pose_grads
        shape_p -= step_size * shape_grads
        expr_p -= step_size * expr_grads
        params = (shape_p, pose_p, expr_p)

        print('iter: ', i)
        
    if show:
        print('truth: ', truth)
        print('predict: ', get_coords_from_params(*params))
        draw_and_show_points(get_coords_from_params(*params))
        draw_and_show_points(truth)
    return shape_p, pose_p, expr_p

def test_extract():
    current_path = os.path.dirname(os.path.abspath(__file__))
    registry_path = os.path.join(current_path, 'registry')
    
    dataset = load(20000, 1, os.path.join(current_path, registry_path, 'dataset'))
    
    el = dataset[0][1][0]
    el = torch.stack([torch.Tensor([th[0], th[1]]) for th in el])
    el = map_points(el)

    print(el.shape)
    extract(el, show=True)

if __name__ == '__main__':
    test_extract()