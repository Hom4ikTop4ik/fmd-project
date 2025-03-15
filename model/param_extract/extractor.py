# shape_params 100
# pose_params 6
# expression_params 50
import torch

def get_coords_from_params(shape, pose, expression):
    '''
    THIS IS PLACEHOLDER
    for function that gives 2D coordinates from FLAME model based on 
    shape, pose and expression parameters
    '''
    return torch.zeros(68, 2)


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
    grad = torch.zeros(params[idparam].shape[0])
    for i in range(params[idparam].shape[0]):
        oldval = params[idparam][i]
        params[idparam][i] += precise
        newloss = lossRMSE(truth, get_coords_from_params(*params))
        params[idparam][i] = oldval
        grad[i] = (init_loss + newloss) / precise
    return grad

def calc_grads( truth: torch.Tensor, flag_shape, flag_pose, flag_expr,
                params, precise: float):
    '''
    Function calculates gradients for groups of parameters 
    that are specified by flags
    '''
    shape_p, pose_p, expr_p = params
    init_loss = lossRMSE(truth, get_coords_from_params(*params))
    shape_grads = torch.zeros(100)
    pose_grads = torch.zeros(6)
    expr_grads = torch.zeros(50)

    if flag_shape:
        shape_grads = calc_grad(params, truth, init_loss, 0, precise)
    if flag_pose:
        pose_grads = calc_grad(params, truth, init_loss, 1, precise)
    if flag_expr:
        expr_grads = calc_grad(params, truth, init_loss, 2, precise)

    return shape_grads, pose_grads, expr_grads
    

def extract(truth: torch.Tensor):
    '''
    Function that user gradient descent to find 
    FLAME parameters, for which FLAME face will be most similar to the truth

    for now it finds only pose, for test.
    '''
    shape_p = torch.zeros(100) # maybe later we can put zeros here 
    pose_p = torch.zeros(6)
    expr_p = torch.zeros(50)
    params = (shape_p, pose_p, expr_p)
    conv_iters = 100 # count of iteration until convergence
    step_size = 0.01
    for i in range(conv_iters):
        shape_grads, pose_grads, expr_grads = calc_grads(
            truth, False, True, False, params, 0.0001
        )
        pose_p -= step_size * pose_p
    return shape_p, pose_p, expr_p


extract()