from PIL import Image, ImageDraw
import torch

def get_mean_coords(landmarks_list: list, tensor: torch.Tensor):
    x = torch.zeros(tensor.shape[0], dtype=torch.float32)
    y = torch.zeros(tensor.shape[0], dtype=torch.float32)
    for id in landmarks_list:
        x += tensor[:, id, 1]
        y += tensor[:, id, 2]
    x /= len(landmarks_list)
    y /= len(landmarks_list)
    return torch.stack([x, y])


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