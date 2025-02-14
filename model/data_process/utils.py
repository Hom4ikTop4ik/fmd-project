import torch

__all__ = ['make_highlighter', 'face_coord']

def get_mean_coords(landmarks_list: list, tensor: torch.Tensor):
    x = torch.zeros(tensor.shape[0], dtype=torch.float32)
    y = torch.zeros(tensor.shape[0], dtype=torch.float32)
    for id in landmarks_list:
        x += tensor[:, id, 1]
        y += tensor[:, id, 2]
    x /= len(landmarks_list)
    y /= len(landmarks_list)
    return torch.stack(torch.Tensor([x, y]))

def face_coord(coordstensor: torch.Tensor):
    return coordstensor.mean(1)[:, 0:2]


def make_highlighter(imgsize: tuple, h_rect: int, w_rect: int, device):
    def highlighter(coordstensor: torch.Tensor):
        out = torch.zeros([coordstensor.shape[0], imgsize, imgsize])
        for id, coords in enumerate(coordstensor):
            means = coords.mean(0)
            bord = 8
            x_mean, y_mean = means[0], means[1]
            left = int(max(x_mean * imgsize - w_rect, 0))
            right = int(x_mean * imgsize + w_rect)
            top = int(max(y_mean * imgsize - h_rect, 0))
            bottom = int(y_mean * imgsize + h_rect)
            print('coords', left, right, top, bottom, x_mean * imgsize - w_rect)
            out[id, top + bord:bottom - bord, left + bord:right - bord] = 0.5
            out[id, top:bottom, left:right] = 1
        
        return out.unsqueeze(1).to(device)
    return highlighter


