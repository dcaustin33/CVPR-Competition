"""
Script to assist in loading data and training FFAnet and FasterRCNN
"""



import torch
from torchvision import transforms, datasets
import torchvision

# Import common neural network API in pytorch
import torch.nn as nn
import torch.nn.functional as F

# Import optimizer related API
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Check device, using gpu 0 if gpu exist else using cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


from matplotlib import image
from PIL import Image
from torchvision import transforms as tfs
import os
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from mean_average_precision import MetricBuilder
import copy

x_size = 224
y_size = 224


#visualize a predicted image and returns to scale if scale is set
def visualize_predicted_image_tensor(im, coords, scale = None, normalized = False):
  if normalized:
    mean = np.array([0.64, 0.6, 0.58])
    std = np.array([0.14,0.15, 0.152])
    im = tfs.Normalize(mean=(-mean / std).tolist(),std= (1/std).tolist())(im)
  #im_height, im_width = image.shape[:2]
  if scale is not None:
    x = int(scale[1][0])
    y = int(scale[1][1])
    im = tfs.Resize((x, y))(im)
    coords = coords * scale[0]

  fig, ax = plt.subplots()
  ax.imshow(np.transpose(im.numpy(), (1, 2, 0)))

  for coord in coords:
    xmin, ymin, xmax, ymax = coord
    height = ymax - ymin
    width = xmax - xmin
    # Blue color in BGR
    color = (255, 0, 0)
      
    # Line thickness of 2 px
    thickness = 2

    rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='b', facecolor='none')
    ax.add_patch(rect)

  plt.show()

#note we are using 224 x 224 for training and the original images
#sets up the data loader dataset for the Dry Run data
class CustomDatasetDryRun(Dataset):
    def __init__(self, path_images, path_labels):
        self.images = sorted(os.listdir(path_images))
        self.labels = sorted(os.listdir(path_labels))
        self.path_images = path_images
        self.all_boxes = []
        self.all_labels = []
        self.scale = {}

        #extract the labels and load into memory
        for i, name in enumerate(self.labels):
          with open(path_labels + name) as f:
            lines = f.readlines()
            labels = np.zeros((12, 1))
            inter_boxes = np.zeros((12, 4))
            for line, l in enumerate(lines):
              boxes = []
              aspects = l.split()
              labels[line, :] = 1
              boxes = aspects[1 : 5]
              boxes = self.label_scaling(boxes)
              boxes = [int(b) for b in boxes]
              inter_boxes[line, :] = boxes

          self.all_boxes.append(inter_boxes)
          self.all_labels.append(labels.reshape(-1))

        self.all_boxes = torch.tensor(np.array(self.all_boxes)).float()
        self.all_labels = torch.tensor(np.array(self.all_labels), dtype = torch.int64)

        #takes the normalization numbers from ffanet
        self.compose = tfs.Compose([tfs.ToTensor()]) #, tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])])


    def __len__(self):
        return len(self.labels)

    def get_scale(self, idx):
      name = self.images[idx]
      im = Image.open(self.path_images + name)
      im = self.compose(im)
      shape = im.shape
      return [np.array([shape[2] / x_size, shape[1] / y_size, shape[2] / x_size, shape[1] / y_size]), (shape[1], shape[2])]

    def label_scaling(self, box):
      x_ = x_size / 1824
      y_ = y_size / 750
      box[0] = float(box[0]) * x_
      box[1] = float(box[1]) * y_
      box[2] = float(box[2]) * x_
      box[3] = float(box[3]) * y_
      return box
    
    def reverse_scaling(self, idx, boxes):
      return boxes * self.scale[idx]


    def __getitem__(self, idx):
      name = self.images[idx]
      im = Image.open(self.path_images + name)
      im = self.compose(im)
      shape = im.shape
      self.scale[idx] = [np.array([shape[2] / y_size, shape[1] / x_size, shape[2] / y_size, shape[1] / x_size]), (shape[1], shape[2])]
      im = tfs.Resize((x_size, y_size))(im)

      target = [{}]
      target[0]['boxes'] = self.all_boxes[idx]
      target[0]['labels'] = self.all_labels[idx]
      mask = (target[0]['labels'][:] != 0).view(-1)
      target[0]['boxes'][mask == False] = target[0]['boxes'][mask == False] + torch.tensor([0, 0, .1, .1]).float()

      target = copy.deepcopy(target)
      return im, target, mask

#note we are using 224 x 224 for training and the original images
#sets up the custom loader dataset for the training data
class CustomDatasetTrain(Dataset):
    def __init__(self, path_images, path_labels):
        self.images = sorted(os.listdir(path_images))
        self.labels = sorted(os.listdir(path_labels))
        self.path_images = path_images
        self.all_boxes = []
        self.all_labels = []
        self.scale = {}

        #extract the labels and load into memory
        for i, name in enumerate(self.labels):
          with open(path_labels + name) as f:
            lines = f.readlines()
            labels = np.zeros((12, 1))
            inter_boxes = np.zeros((12, 4))
            for line, l in enumerate(lines):
              boxes = []
              aspects = l.split()
              labels[line, :] = 1
              boxes = aspects[1 : 5]
              boxes = self.label_scaling(boxes)
              boxes = [int(b) for b in boxes]
              inter_boxes[line, :] = boxes

          self.all_boxes.append(inter_boxes)
          self.all_labels.append(labels.reshape(-1))

        self.all_boxes = torch.tensor(np.array(self.all_boxes)).float()
        self.all_labels = torch.tensor(np.array(self.all_labels), dtype = torch.int64)

        #takes the normalization numbers from ffanet
        self.compose = tfs.Compose([tfs.ToTensor()])


    def __len__(self):
        return len(self.labels)

    def get_scale(self, idx):
      name = self.images[idx]
      im = Image.open(self.path_images + name)
      im = self.compose(im)
      shape = im.shape
      y_prime = y_size

      x_reverse = shape[2] / x_size
      y_reverse = (shape[1] * .5) / y_size
      return [np.array([x_reverse, y_reverse, x_reverse, y_reverse]), (shape[1] / 2, shape[2])]

    def label_scaling(self, box):

      x_ = x_size / 1824
      y_ = y_size / 750
      box[0] = float(box[0]) * x_
      box[1] = float(box[1]) * y_
      box[2] = float(box[2]) * x_
      box[3] = float(box[3]) * y_

      return box


    def __getitem__(self, idx):
      name = self.images[idx]
      im = Image.open(self.path_images + name)
      im = self.compose(im)
      shape = im.shape
      x_reverse = shape[2] / x_size
      y_reverse = (shape[1] * .5) / y_size
      self.scale[idx] = [np.array([x_reverse, y_reverse, x_reverse, y_reverse]), (shape[1] / 2, shape[2])]
      im = tfs.Resize((x_size * 2, y_size))(im)

      im = im[:, :x_size, :] #just takes the hazy portion
      target = [{}]
      target[0]['boxes'] = self.all_boxes[idx]
      target[0]['labels'] = self.all_labels[idx]
      mask = (target[0]['labels'][:] != 0).view(-1)
      target[0]['boxes'][mask == False] = target[0]['boxes'][mask == False] + torch.tensor([0, 0, .1, .1]).float()
      target = copy.deepcopy(target)
      return im, target, mask







#sets a threshold to prune boxes that do not have the desired threshold of having an object within
def process_results(result, threshold = .5):
  box_ind = torchvision.ops.batched_nms(result[0]['boxes'], result[0]['scores'].view(-1), result[0]['labels'], .01).cpu()

  result[0]['scores'] = result[0]['scores'].to('cpu')
  result[0]['boxes'] = result[0]['boxes'].to('cpu')
  result[0]['labels'] = result[0]['labels'].to('cpu')

  result[0]['boxes'] = result[0]['boxes'][box_ind]
  result[0]['scores'] = result[0]['scores'][box_ind]
  mask = (result[0]['scores'] >= threshold)

  result[0]['boxes'] = result[0]['boxes'][mask]
  result[0]['scores'] = result[0]['scores'][mask]
  return result




import copy


#trains the RCNN with self explanatory parameters. Loss weights balance what to weight classifier vs regresion, bs objectness loss
def train_eval_rcnn(data, model, epochs, ffa_net = None, optimizer = None, loss_weights = [1, 1, 1], train = False, lr = .000001, data_augmentation = True):
    train_data = DataLoader(data, batch_size=1, shuffle=True)
    total_classifier = 0
    total_reg = 0
    total_object = 0

    #training
    model = model.to(device)
    model.train()


    if optimizer:
        d_optimizer = optimizer
    else:
        d_optimizer = optim.Adam(model.parameters(), lr = lr)

    d_optimizer.zero_grad()

    for epoch in range(epochs):
        total_classifier = 0
        total_reg = 0
        total_object = 0

        for i, data in enumerate(train_data):

            im, target, mask = data
            im = im.to(device)
            im_before = im
            #copys the old target for rotation during data augmentation 
            old_target = copy.deepcopy(target)
            
            if train and data_augmentation:
                #applies data augmentation with 70% prob
                rand = np.random.randint(0, 10)
                if rand > 2:
                    im, target = data_aug(im, target, mask)

            if ffa_net:
                im = tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(im)
                im = ffa_net(im)
                im = im.clamp(0, 1)

            target = target
            target[0]['boxes'] = target[0]['boxes'][0][mask[0]].to(device)
            target[0]['labels'] = target[0]['labels'][0][mask[0]].to(device)

            results = model(im, target)
            weight1, weight2, weight3 = loss_weights[0], loss_weights[1], loss_weights[2]
            loss = weight1 * results['loss_box_reg'] + weight2 * results['loss_classifier'] + weight3 * results['loss_objectness']
            total_classifier += results['loss_classifier'].item()
            total_reg += results['loss_box_reg'].item()
            total_object += results['loss_objectness'].item()
            if train:
                loss.backward()
                d_optimizer.step()
                d_optimizer.zero_grad()
            #puts to before the augmentation
            target = old_target

        print(total_classifier / i, total_reg / i, total_object / i)
    return model, d_optimizer


#prints the rcnn pic with the predicted bounding box
def print_rcnn_pic(model, idx, data, ffa_net = None, threshold = .5):
    with torch.no_grad():
        im, target, mask = data.__getitem__(idx)
        im = im.to(device)
        model = model.to(device)
        if ffa_net:
            im = tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(im)
            im = ffa_net(im.unsqueeze(0))[0]
            imm = im.clamp(0, 1)

        model.eval()
        result = model(im.unsqueeze(0))
        result = process_results(result, threshold)


    visualize_predicted_image_tensor(im.cpu(), result[0]['boxes'], data.get_scale(idx))

#writes to the dry run directory after prediciting
def write_dry_run(dry_data, rcnn, output_dir, ffa_net = None, threshold = .5):
    model = rcnn
    if type(model) == list:
        model[0].eval()
        model[1].eval()
    else:
        model.eval()
    with torch.no_grad():
        for i in range(dry_data.__len__()):
            im, target, mask = dry_data.__getitem__(i)

            #I believe this should be the name
            filename = dry_data.images[i].split('.')[0] + '.txt'
            new_file = os.path.join(output_dir, filename)
            im = im.to(device)
            
            if ffa_net:
                im = tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(im)
                im = ffa_net(im.unsqueeze(0))[0]
                im = im.clamp(0, 1)
                
                
            if type(model) == list:
                results = [{}]
                result1 = model[0](im.unsqueeze(0))
                result2 = model[1](im.unsqueeze(0))

                results[0]['boxes'] = torch.tensor(np.vstack((result1[0]['boxes'].cpu(), result2[0]['boxes'].cpu())))
                results[0]['labels'] = torch.tensor(np.concatenate((result1[0]['labels'].cpu(), result2[0]['labels'].cpu())))
                results[0]['scores'] = torch.tensor(np.concatenate((result1[0]['scores'].cpu(), result2[0]['scores'].cpu())))
            else:          
                results = model(im.unsqueeze(0))

            results = process_results(results, threshold)
            results[0]['boxes'] = results[0]['boxes'] * dry_data.get_scale(i)[0]
            scores = results[0]['scores']
            coords = results[0]['boxes']
            with open(new_file, 'w') as f:
                for loc, coord in enumerate(coords):
                    coord = coord.numpy()
                    f.write('vehicle ' + str(coord[0]) + " " + str(coord[1]) + " " + str(coord[2]) + ' ' + str(coord[3]) + " " + str(scores[loc].numpy()) + "\n")





"""
Custom Functions for the ffa net training portion
"""
#creates a custom dataset for the FFA train
class CustomDatasetFFA(Dataset):
  def __init__(self, path_images):
      self.images = sorted(os.listdir(path_images))
      self.path_images = path_images
      self.scale = {}

      #takes the normalization numbers from ffanet
      self.compose = tfs.Compose([tfs.ToTensor(), tfs.Resize((x_size * 2, y_size)), 
                     tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])])


  def __len__(self):
      return len(self.images)

  def get_scale(self, idx):
    name = self.images[idx]
    im = Image.open(self.path_images + name)
    im = self.compose(im)
    shape = im.shape
    x_reverse = shape[2] / x_size
    y_reverse = (shape[1] * .5) / y_size
    return [np.array([x_reverse, y_reverse, x_reverse, y_reverse]), (shape[1] / 2, shape[2])]


  def unnormalize(self, image):
    mean=[0.64, 0.6, 0.58]
    std=[0.14,0.15, 0.152]
    tfs.Normalize((-mean / std).tolist(), (1.0 / std).tolist())(image)
    return image

  def __getitem__(self, idx):
    name = self.images[idx]
    im = Image.open(self.path_images + name)
    label = tfs.ToTensor()(im)
    label = tfs.Resize((x_size * 2, y_size))(label)
    im = self.compose(im)
    shape = im.shape
    x_reverse = shape[2] / x_size
    y_reverse = (shape[1] * .5) / y_size
    self.scale[idx] = [np.array([x_reverse, y_reverse, x_reverse, y_reverse]), (shape[1] / 2, shape[2])]

    im = im[:, :x_size, :] #just takes the hazy portion
    label = label[:, x_size:, :] #just takes the clean portion

    return im, label

#creates a custom data set for the ffa on the dry run
class CustomDatasetDryFFA(Dataset):
  def __init__(self, path_images):
      self.images = sorted(os.listdir(path_images))
      self.path_images = path_images
      self.scale = {}

      #takes the normalization numbers from ffanet
      self.compose = tfs.Compose([tfs.ToTensor(), tfs.Resize((x_size, y_size)), 
                     tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])])


  def __len__(self):
      return len(self.images)

  def get_scale(self, idx):
    name = self.images[idx]
    im = Image.open(self.path_images + name)
    im = self.compose(im)
    shape = im.shape
    return [np.array([shape[2] / y_size, shape[1] / x_size, shape[2] / y_size, shape[1] / x_size]), (shape[1], shape[2])]


  def unnormalize(self, image):
    mean=[0.64, 0.6, 0.58]
    std=[0.14,0.15, 0.152]
    tfs.Normalize((-mean / std).tolist(), (1.0 / std).tolist())(image)
    return image

  def __getitem__(self, idx):
    name = self.images[idx]
    im = Image.open(self.path_images + name)
    im = self.compose(im)
    shape = im.shape
    self.scale[idx] = [np.array([shape[2] / y_size, shape[1] / x_size, shape[2] / y_size, shape[1] / x_size]), (shape[1], shape[2])]

    im = im[:, :x_size, :] #just takes the hazy portion

    return im

#trains the ffa with self explanatory parameters
def train_ffa(data, model, epochs, lr = .00001, optimizer = None):
    train_data = DataLoader(data, batch_size=2, shuffle=True)

    #training
    model = model.to(device)
    model.train()

    total_loss = 0
    loss_f = nn.L1Loss()
    if optimizer:
        d_optimizer = optimizer
    else:
        d_optimizer = optim.Adam(model.parameters(), lr = lr)

    d_optimizer.zero_grad()

    for epoch in range(epochs):
        total_loss = 0

        for i, data in enumerate(train_data):
            im, label = data
            im = im.to(device)
            label = label.to(device)

            results = model(im)
            loss = loss_f(results, label)
            total_loss += loss.item()

            loss.backward()
            d_optimizer.step()
            d_optimizer.zero_grad()
        print(total_loss / i)
    model.train()
    return model, optimizer

#evals the ffa with the given data set
def eval_ffa(data, model, epochs):
    model.eval()
    with torch.no_grad():
        train_data = DataLoader(data, batch_size=1, shuffle=True)

        #training
        model = model.to(device)
        model.train()

        total_loss = 0
        mseLoss = nn.MSELoss()


        for epoch in range(epochs):
            total_loss = 0

            for i, data in enumerate(train_data):
                im, label = data
                im = im.to(device)
                label = label.to(device)

                results = model(im)
                loss = mseLoss(results, label)
                total_loss += loss.item()
            print(total_loss / i)
        model.train()
        return (total_loss / i)


#prints a cleaned ffa pic with its counterpart    
def print_ffa_pic(idx, ffa_data, net):
    label = None
    im = ffa_data.__getitem__(idx)
    print(len(im))
    if len(im) == 2:
        im, label = im
    visualize_predicted_image_tensor(im.cpu(), [], normalized = True)

    with torch.no_grad():
        im = im.to(device)
        im = im[:, :224, :]
        net = net.to(device)
        result = net(im.unsqueeze(0))[0]
        visualize_predicted_image_tensor(result.cpu(), [])
    if label != None:
        visualize_predicted_image_tensor(label.cpu(), [])
        
def read_box(stem, labels_):
    labels = sorted(os.listdir(stem))
    total = []
    for p in labels:
        if p == '.DS_Store': continue
        with open(stem + p) as f:
            lines = f.readlines()
            current = []
            for i in lines:
                i = i.split()[1:]
                i = i[:4] + [0] + i[4:]
                if labels_:
                    i = i + [0, 0]
                i = [float(num) for num in i]
                current.append(i)
            total.append(current)
    return total
#function to compute the mAP with the dry run data set
def getmAP(label_path, pred_path):
    preds = read_box(pred_path, False)
    real = read_box(label_path, True)
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=1)
    for i, pred in enumerate(preds):
        pred = np.array(pred)
        label = np.array(real[i])
        metric_fn.add(pred, label)
    print(f"VOC PASCAL mAP in all points: {metric_fn.value(iou_thresholds=0.5)['mAP']}")
    return metric_fn.value(iou_thresholds=0.5)['mAP']
    
    
#rotates the boxes for data augmentation    
def rotate_box(box, angle):
    angle *= np.pi /180
    cos = np.cos(-angle)
    sin = np.sin(-angle)
    
    #rotating around 124 124
    coords1 = box[:2] - torch.tensor([112, 112])
    coords2 = box[2:] - torch.tensor([112, 112])

    coords1[0], coords1[1] = cos * coords1[0] - sin * coords1[1] + 112, sin * coords1[0] + cos * coords1[1] + 112
    coords2[0], coords2[1] = cos * coords2[0] - sin * coords2[1] + 112, sin * coords2[0] + cos * coords2[1] + 112
    
    box[0] = min(coords1[0], coords2[0])
    box[2] = max(coords1[0], coords2[0])
    box[1] = min(coords1[1], coords2[1])
    box[3] = max(coords1[1], coords2[1])
    return box

#applies data agumentation
def data_aug(im, labels, mask):
    
    #applies a gaussian blur
    rand = np.random.randint(0, 30)
    if rand % 2 == 0: rand += 1
    im = tfs.GaussianBlur(rand)(im)
    
    #applies a random sharpness
    rand = np.random.randint(0, 10)
    im = tfs.RandomAdjustSharpness(10, p=.5)(im)
    
    #random solarization
    im = tfs.RandomSolarize(.8, p=0.2)(im)
    
    #rotates every time
    rand = np.random.choice([0, 90, 180, 270])
    im = tfs.functional.rotate(im, int(rand), fill = 1)
    for i, lab in enumerate(labels[0]['boxes'][mask]):
        labels[0]['boxes'][0][i] = rotate_box(lab, rand)
        
    im = tfs.ColorJitter(brightness=1, contrast=1, saturation=5, hue=.2)(im)
        
    return im, labels
        
#trains all parameters of FFA_net and faster RCNN at the same time    
def train_end_to_end(data, model, epochs, ffa_net, optimizer = None, loss_weights = [1, 1, 1], train = False, lr = .000001, data_augmentation = True):
    train_data = DataLoader(data, batch_size=1, shuffle=True)
    total_classifier = 0
    total_reg = 0
    total_object = 0

    #training
    model = model.to(device)
    model.train()


    if optimizer:
        d_optimizer = optimizer
    else:
        params = list(model.parameters()) + list(ffa_net.parameters())
        d_optimizer = optim.Adam(params, lr = lr)

    d_optimizer.zero_grad()

    for epoch in range(epochs):
        total_classifier = 0
        total_reg = 0
        total_object = 0

        for i, data in enumerate(train_data):

            im, target, mask = data
            im = im.to(device)
            im_before = im
            old_target = copy.deepcopy(target)
            
            if train and data_augmentation:
                #applies data augmentation with 70% prob
                rand = np.random.randint(0, 10)
                if rand > 2:
                    im, target = data_aug(im, target, mask)

            im = tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(im)
            im = ffa_net(im)
            im = im.clamp(0, 1)

            target = target
            target[0]['boxes'] = target[0]['boxes'][0][mask[0]].to(device)
            target[0]['labels'] = target[0]['labels'][0][mask[0]].to(device)

            results = model(im, target)
            weight1, weight2, weight3 = loss_weights[0], loss_weights[1], loss_weights[2]
            loss = weight1 * results['loss_box_reg'] + weight2 * results['loss_classifier'] + weight3 * results['loss_objectness']
            total_classifier += results['loss_classifier'].item()
            total_reg += results['loss_box_reg'].item()
            total_object += results['loss_objectness'].item()
            if train:
                loss.backward()
                d_optimizer.step()
                d_optimizer.zero_grad()
            #puts to before the augmentation
            target = old_target

        print(total_classifier / i, total_reg / i, total_object / i)
    return model, ffa_net, d_optimizer
