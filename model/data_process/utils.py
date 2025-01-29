from PIL import Image, ImageDraw
import torch

__all__ = ['make_highlighter']

def get_mean_coords(landmarks_list: list, tensor: torch.Tensor):
    x = torch.zeros(tensor.shape[0], dtype=torch.float32)
    y = torch.zeros(tensor.shape[0], dtype=torch.float32)
    for id in landmarks_list:
        x += tensor[:, id, 1]
        y += tensor[:, id, 2]
    x /= len(landmarks_list)
    y /= len(landmarks_list)
    return torch.stack(torch.Tensor([x, y]))

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





def show_tensor(tensor: torch.Tensor, landmarks = None, nolandmarks=False, no_z = False):
    # Make sure tensors are on CPU and detached from grad
    image = tensor.cpu().detach()
    
    #image /= image.mean()
    # Scale to 0-255 range
    image = (image * 255).clamp(0, 255)
    # Convert to numpy and correct data type
    image = image.numpy().astype('uint8')
    # If tensor is [C,H,W], convert to [H,W,C]
    if len(image.shape) == 3:
        image = image.transpose(1, 2, 0)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    
    # Get image dimensions
    width, height = pil_image.size
    
    if(nolandmarks == True):
        return pil_image
    landmarks = landmarks.cpu().detach()
    # Draw each landmark
    for i in range(68):
        # Get coordinates (scale from 0-1 to image dimensions)
        x = int(landmarks[i, 0].item() * width)
        y = int(landmarks[i, 1].item() * height)
        z = 0.0
        if no_z == False:
            z = landmarks[i, 2].item()
        
        # Draw point (red circle)
        radius = 2
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill='white')
        
        # Draw value next to point
        draw.text((x+5, y-5), f'{z:.2f}', fill='white')
    
    return pil_image