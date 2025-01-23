# Converted from notebook: mainnb.ipynb

# Cell
import data_loader

import torch
import torch.nn as nn

import random
import normalizer
from PIL import Image, ImageDraw, ImageFont

import models
import modelutils

# Cell


# Cell
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))




train = data_loader.load("I:/NSU/CV/tests/torch/data/train/coords",
                         "I:/NSU/CV/tests/torch/data/train/images", 
                        firstn = 7000, batchSize = 16, shuffle = True)



scaler = normalizer.MinMaxNormalizer()
scaler.fit([y for _, y in train])

print("Number of batches:", len(train))
for x, y in train:
    print(x.shape, y.shape)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("devise is: ", device)



# Cell
class CustomLoss(nn.Module):
    def __init__(self, vertlist, reverse):
        super().__init__()
        self.weight = weight
    
    def forward(self, predictions, targets):
        # You can implement any custom loss calculation here
        element_wise_loss = torch.abs(predictions - targets)
        # You can add weights, combine losses, or add regularization terms
        weighted_loss = element_wise_loss * self.weight
        return torch.mean(weighted_loss)

# Cell
train[0][0].shape

# Cell
class SelectiveRMSELoss(nn.Module):
    def __init__(self, pointlist, reverse, device, l2_lambda = 0.0):
        super().__init__()
        self.reverse = reverse
        self.pointlist = pointlist
        self.device = device
        self.l2_lambda = l2_lambda
        
    def forward(self, x, y, parameters = None):
        ls = (x-y)**2
        losslist = []
        if(self.reverse):
            self.pointlist = list(set(range(68)) - set(self.pointlist))
        sm = torch.tensor(0.0).to(device)
        k = torch.tensor(1.0).to(device)
        for i in range(x.shape[0]):
            for j in self.pointlist:
                sm += (ls[i][j].mean())
                k += 1
        if parameters is None:
            return torch.sqrt(sm / float(k))
        
        pk = 0.0
        smp = 0.0
        if parameters is not None:
            for param in parameters:
                smp += (param**2).mean()
                pk += 1

        return torch.sqrt(sm / float(k)) + (smp / float(pk)) * self.l2_lambda

# Cell
mouth_pointlist = [44, 7, 33, 14, 2, 31, 49, 15, 42, 32, 9, 51, 38, 61,
    18, 23, 12, 47, 67, 1, 2]
mouth_boundaries = [7, 14, 15, 67]
eye_L_pointlist = [62, 65, 0, 13, 34, 64]
eye_R_pointlist = [16, 36, 54, 55, 53, 63]

# Cell



# Cell
def get_mean_coords(landmarks_list, tensor):
    x = torch.zeros(tensor.shape[0]).to(device)
    y = torch.zeros(tensor.shape[0]).to(device)
    for id in landmarks_list:
        x += tensor[:,id,0]
        y += tensor[:,id,1]
    x /= len(landmarks_list)
    y /= len(landmarks_list)
    return x, y


def crop_mouth(image_tensor_array, landmarks_tensor_array, eye_L_pointlist, eye_R_pointlist):
    # Get mouth landmarks

    x_l_ar, y_l_ar = get_mean_coords(eye_L_pointlist, landmarks_tensor_array)
    x_r_ar, y_r_ar = get_mean_coords(eye_R_pointlist, landmarks_tensor_array)

    for image_tensor, x_l, y_l, x_r, y_r in zip(image_tensor_array, x_l_ar, y_l_ar, x_r_ar, y_r_ar):
        
        # mouth_x = (x_l + x_r) / 2 + (y_r - y_l)
        # eyl = torch.sqrt((y_r - y_l)**2 + (x_r - x_l)**2)

        # eyl_s = (torch.sqrt(1 - ((y_r - y_l)**2) / eyl**2)) * eyl

        # mouth_y = (y_l + y_r) / 2 + eyl_s

        mouth_x = (x_l + x_r) / 2
        mouth_y = (y_l + y_r) / 2
        
        size = image_tensor.shape[2]

        x1 = int(mouth_x * size - 100)
        x2 = int(mouth_x * size + 100)
        y1 = int(mouth_y * size - 64)
        y2 = int(mouth_y * size + 64)
        
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x1 > size:
            x1 = size
        if y1 > size:
            y1 = size

        # Crop image
        image_tensor = image_tensor[:, y1:y2, x1:x2]
        # Get bounding box of mouth
    return image_tensor


# Cell
class MouthPointsDetector(nn.Module):
    def __init__(self, device, eyes_detector, eye_L_pointlist, eye_R_pointlist):
        super(MouthPointsDetector, self).__init__()
        self.eyes_detector = eyes_detector.eval()
        # Freeze the eyes detector parameters
        for param in self.eyes_detector.parameters():
            param.requires_grad = False
        self.last_detector_size = 8
        self.eye_L_pointlist = eye_L_pointlist
        self.eye_R_pointlist = eye_R_pointlist
        
        self.conv1 = nn.Conv2d(3, 5, 3, padding = 1).to(device)
        self.pool = nn.MaxPool2d(2, 2).to(device)
        self.conv2 = nn.Conv2d(5, 5, 3, padding = 1).to(device)
        self.conv3 = nn.Conv2d(5, 5, 3, padding = 1).to(device)
        self.conv4 = nn.Conv2d(5, 5, 3, padding = 1).to(device)
        self.conv5 = nn.Conv2d(5, 6, 3, padding = 1).to(device)
        self.conv6 = nn.Conv2d(6, self.last_detector_size, 3, padding = 1).to(device)
        fcsize = 612
        
        self.adpool_size = 54, 30
        self.adpool = nn.AdaptiveAvgPool2d(self.adpool_size).to(device) 

        self.fc1 = nn.Linear(self.last_detector_size*self.adpool_size[0]*self.adpool_size[1],
                             fcsize).to(device)
        self.fc_list = nn.ModuleList([nn.Linear(fcsize, fcsize).to(device) for _ in range(12)])
        self.prelast = nn.Linear(fcsize, fcsize).to(device)
        self.fc_last = nn.Linear(fcsize, 3 * 68).to(device)
        self.act = nn.ReLU().to(device)
        self.sigm = nn.Tanh().to(device)
    
    def forward(self, x, needshow = False):
        # Input: [batch, 3, H, W]
        ans_with_eyes = self.eyes_detector(x)

        if(needshow):
            xshow = modelutils.show_tensor(x[0], ans_with_eyes[0], nolandmarks=False)
            xshow.show()
        with torch.no_grad():
            x = crop_mouth(x, ans_with_eyes, self.eye_L_pointlist, self.eye_R_pointlist)
        
        if(needshow):
            xshow = modelutils.show_tensor(x[0], ans_with_eyes[0], nolandmarks=True)
            xshow.show()

        x = self.act(self.conv1(x))
        
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.pool(self.act(self.conv5(x)))
        x = self.act(self.conv6(x))  
        #print(x.shape)
        
        #x = self.act(self.conv6(x))
        x = self.adpool(x)
        
        x = x.view(-1, self.last_detector_size*self.adpool_size[0]*self.adpool_size[1])       
        x = self.act(self.fc1(x))
        for i in range(len(self.fc_list)):
            x = self.sigm(self.fc_list[i](x))
        x = self.sigm(self.prelast(x))
        x = self.fc_last(x)
        x = x.view(-1, 68, 3)
        return x


# Cell
class EyesDetector(nn.Module):
    def __init__(self, device):
        super(EyesDetector, self).__init__()
        self.last_detector_size = 128 
        self.adpool = nn.AdaptiveAvgPool2d((128, 128)).to(device) 
        self.conv1 = nn.Conv2d(3, 6, 3, padding = 1).to(device)
        self.pool = nn.MaxPool2d(2, 2).to(device)
        self.conv2 = nn.Conv2d(6, 9, 3, padding = 1).to(device)
        self.conv3 = nn.Conv2d(9, 20, 3, padding = 1).to(device)
        self.conv4 = nn.Conv2d(20, self.last_detector_size, 3, padding = 1).to(device)
        fcsize = 256
        self.fc1 = nn.Linear(self.last_detector_size*16*16, fcsize).to(device)
        self.fc_list = nn.ModuleList([nn.Linear(fcsize, fcsize).to(device) for _ in range(3)])
        self.prelast = nn.Linear(fcsize, fcsize).to(device)
        self.fc_last = nn.Linear(fcsize, 3 * 68).to(device)
        self.act = nn.ReLU().to(device)
        self.sigm = nn.Sigmoid().to(device)
    
    def forward(self, x, needshow = False):
        # Input: [batch, 3, H, W]
        x = self.adpool(x)
        x = self.pool(self.act(self.conv1(x)))
        if(needshow):
            xshow = modelutils.show_tensor(x[0:2], landmarks=None, nolandmarks=True)
            xshow.show()
        x = self.pool(self.act(self.conv2(x)))
        x = self.act(self.conv3(x))  
        x = self.pool(self.act(self.conv4(x)))
        x = x.view(-1, self.last_detector_size*16*16)       
        x = self.act(self.fc1(x))
        for i in range(len(self.fc_list)):
            x = self.act(self.fc_list[i](x))
        x = self.act(self.prelast(x))
        x = self.fc_last(x)
        x = x.view(-1, 68, 3)
        return x

# Cell
modelEyeDetector = EyesDetector(device)
state = torch.load('eyeDetector.pth')
modelEyeDetector.load_state_dict(state['model_state_dict'])
modelEyeDetector.to(device)
modelEyeDetector.eval()

# Cell
model = MouthPointsDetector(device, modelEyeDetector, 
                            eye_L_pointlist, eye_R_pointlist).to(device)

#model = EyesDetector(device)

criterion = SelectiveRMSELoss(mouth_pointlist, False, device, l2_lambda = 0.05)
#criterion = SelectiveRMSELoss(eye_L_pointlist + eye_R_pointlist, False, device, l2_lambda = 0.05)

learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# learning loop
epoch_loss = 0
for epoch in range(1):
    epoch_loss = 0
    step_loss = 0
    random.shuffle(train)
    for batch_idx, (inputs, answers) in enumerate(train):
        needshow = torch.tensor(False).to(device)
        if(batch_idx % 100 == 0):
            needshow = True
        inputs = inputs.to(device)
        answers = answers.to(device)
        answers = scaler.transform(answers)
        outputs = model(inputs, needshow)
        outputs = scaler.inverse_transform(outputs)
        # if(needshow):
        #     xshow = modelutils.show_tensor(inputs[0], outputs[0], nolandmarks=False)
        #     xshow.show()

        loss = criterion(outputs, answers,  model.parameters())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() 
        
        if batch_idx % 5 == 1:
            print(f'Batch {batch_idx}, Loss: {loss.item():.5f}')

        if batch_idx == 1600:
            break
    
    learning_rate /= 10.0
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print(f'shape {inputs.shape}, Epoch {epoch + 1}, Loss: {epoch_loss/len(train):.5f}')


# Cell
# testloop
criterion = SelectiveRMSELoss(eye_L_pointlist + eye_R_pointlist, False, device)
epoch_loss = 0
test = data_loader.load("I:/NSU/CV/tests/torch/data/test/coords",
                        "I:/NSU/CV/tests/torch/data/test/images",  
                        firstn = 2000, batchSize = 16, shuffle = True)
random.shuffle(test)
                        


# Cell
model

# Cell
with torch.no_grad():
    for batch_idx, (inputs, answers) in enumerate(test):
        inputs = inputs.to(device)
        answers = answers.to(device)
        answers = scaler.transform(answers)
        outputs = model(inputs)
        outputs = scaler.inverse_transform(outputs)
        loss = criterion(outputs, answers)
        epoch_loss += loss.item()

    print(f'Test Loss: {epoch_loss/len(test):.4f}')


# Cell
random.shuffle(test)

# Cell


# Cell

with torch.no_grad():
    inputs, answers = test[1]
    inputs = inputs.to(device)
    answers = answers.to(device)
    outputs = modelEyeDetector(inputs)
    outputs = scaler.inverse_transform(outputs)
    # x_l, y_l = get_mean_coords(eye_L_pointlist, outputs)
    # x_r, y_r = get_mean_coords(eye_R_pointlist, outputs)
    # print(x_l[0], y_l[0], x_r[0], y_r[0])
    # outputs[0][1][0] = x_l[0]
    # outputs[0][1][1] = y_l[0]
    # outputs[0][2][0] = x_r[0]
    # outputs[0][2][1] = y_r[0]
    print(outputs.shape, answers.shape)
    img = modelutils.show_tensor(inputs[0], outputs[0])
    img.show()
    imgdlib = modelutils.show_tensor(inputs[0], answers[0])
    imgdlib.show()



# Cell


#Save the model after training
torch.save({
    'model_state_dict': model.state_dict(),
}, 'eyeDetector.pth')

# Cell
class mouthBoundDetector(nn.Module):
    def __init__(self, device):
        super(mouthBoundDetector, self).__init__()
        self.last_detector_size = 128 
        self.adpool = nn.AdaptiveAvgPool2d((128, 128)).to(device) 
        self.conv1 = nn.Conv2d(3, 6, 3, padding = 1).to(device)
        self.pool = nn.MaxPool2d(2, 2).to(device)
        self.conv2 = nn.Conv2d(6, 9, 3, padding = 1).to(device)
        self.conv3 = nn.Conv2d(9, 20, 3, padding = 1).to(device)
        self.conv4 = nn.Conv2d(20, self.last_detector_size, 3, padding = 1).to(device)
        fcsize = 256
        self.fc1 = nn.Linear(self.last_detector_size*16*16, fcsize).to(device)
        self.fc_list = []
        for i in range(3):
            self.fc_list.append(nn.Linear(fcsize, fcsize).to(device))
        self.prelast = nn.Linear(fcsize, fcsize).to(device)
        self.fc_last = nn.Linear(fcsize, 3 * 68).to(device)
        self.act = nn.ReLU().to(device)
        self.sigm = nn.Sigmoid().to(device)
    
    def forward(self, x):
        # Input: [batch, 3, H, W]
        x = self.adpool(x)
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = self.act(self.conv3(x))  
        x = self.pool(self.act(self.conv4(x)))
        x = x.view(-1, self.last_detector_size*16*16)       
        x = self.act(self.fc1(x))
        for i in range(len(self.fc_list)):
            x = self.act(self.fc_list[i](x))
        x = self.act(self.prelast(x))
        x = self.fc_last(x)
        x = x.view(-1, 68, 3)
        return x


# Cell
with torch.no_grad():
    inputs, answers = test[1]
    inputs = inputs.to(device)
    answers = answers.to(device)
    outputs = model(inputs)
    outputs = scaler.inverse_transform(outputs)
    print(outputs.shape, answers.shape)
    img = show_tensor(inputs[0], outputs[0])
    img.show()
    imgdlib = show_tensor(inputs[0], answers[0])
    imgdlib.show()
