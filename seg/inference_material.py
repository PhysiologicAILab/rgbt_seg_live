import datetime
import os
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import joint_transforms_4inputs as joint_transforms
import matplotlib.pyplot as plt
from dataset_materials_rgbt import ImageFolder
import dataset_materials
from misc import AvgMeter, check_mkdir
from models.LSNet_materials import LSNet
from torch.backends import cudnn
import torch.nn.functional as functional
import numpy as np
import cv2
import torch.nn.functional as F
from PIL import Image

cudnn.benchmark = True

torch.manual_seed(2023)
torch.cuda.set_device(3)



VOC_COLORMAP = [0, 0, 0, 200, 128, 128, 128, 128, 200]

test_data = './data/W2P5D2_FINAL'
    


def visualize_prediction1(images, outputs,thermals):
    for kk in range(outputs.shape[0]):
        pred_edge_kk = outputs[kk, :, :, :]
        pred = torch.argmax(pred_edge_kk, 0)
        if len(torch.unique(pred))>2:
            pred = pred.cpu().data.numpy()
            #print(images.shape)
            image = images[kk, :, :, :]
            image = image.cpu().data.numpy()
            #print(image.shape)
            #image = np.transpose(image, (1, 2, 0))
            rgb_img = Image.fromarray(image.astype('uint8')).convert('RGB')
            
            #thermal = thermals
            
            '''
            thermal = thermals[kk, :,:]
            thermal = thermal.cpu().data.numpy()
            #print(np.unique(thermal))
            thermal = thermal * 255
            #print(np.unique(thermal))
            thermal = np.transpose(thermal, (1, 2, 0))
            thermal_img = Image.fromarray(thermal.astype('uint8')).convert('RGB')
            '''
            gt = thermals[kk, :,:].cpu().data.numpy()
            gt_img = Image.fromarray(gt.astype('uint8'))
            gt_img.putpalette(VOC_COLORMAP)
            gt_mask = gt_img.convert('RGBA')
            gt_mask.putalpha(128)

            original_img = Image.fromarray(image.astype('uint8')).convert('RGBA')
            original_img = original_img.resize(gt_mask.size)
            result_image_gt = Image.alpha_composite(original_img, gt_mask)
            
            
            out_img = Image.fromarray(pred.astype('uint8'))
            out_img.putpalette(VOC_COLORMAP)
            mask = out_img.convert('RGBA')
            mask.putalpha(128)

            original_img = Image.fromarray(image.astype('uint8')).convert('RGBA')
            original_img = original_img.resize(mask.size)
            result_image = Image.alpha_composite(original_img, mask)
        
            save_path = './Demo/'
            save_path_mask = './Demo/Masks/'
            save_path_rgb = './Demo/RGB/'
            save_path_thermal = './Demo/GT/'
            
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if not os.path.exists(save_path_mask):
                os.makedirs(save_path_mask)
            if not os.path.exists(save_path_rgb):
                os.makedirs(save_path_rgb)
                
            if not os.path.exists(save_path_thermal):
                os.makedirs(save_path_thermal)
        
            name = '{:02d}_mask.png'.format(kk)
            result_image.save(os.path.join(save_path, name))
            out_img.save(os.path.join(save_path_mask, name))
            rgb_img.save(os.path.join(save_path_rgb, name))
            result_image_gt.save(os.path.join(save_path_thermal, name))





##########################hyperparameters###############################
ckpt_path = './Materials_damage'
exp_name = 'Demo_checkpoint'
args = {
    'iter_num':20000,
    'train_batch_size': 4,
    'last_iter': 0,
    'lr': 5e-5,
    'lr_decay': 0.9,
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'longer_size': 512,
    'crop_size': 640,
    'snapshot': ''
}
##########################data augmentation###############################

joint_transform = joint_transforms.Compose([
    joint_transforms.RandomCrop(args['crop_size'],args['crop_size'])
    #joint_transforms.RandomHorizontallyFlip(),
    #joint_transforms.RandomRotate(10)
])
img_transform = transforms.Compose([
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
target_transform = transforms.ToTensor()

##########################################################################

test_set = ImageFolder(test_data, joint_transform, img_transform, target_transform,args['crop_size'])


test_loader = DataLoader(test_set, batch_size=args['train_batch_size'], num_workers=12, shuffle=True)



def main():
    model = LSNet()
    net = model.cuda().train()
    net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '20000.pth')))
    
    

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    
    train(net)

def train(net):
    curr_iter = args['last_iter']
    while True:
        total_loss_record, loss1_record, loss2_record,loss3_record,loss4_record,loss5_record,loss6_record,loss7_record,loss8_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(),AvgMeter(),AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

        for i, data in enumerate(test_loader):
            
            inputs, thermal, labels,image_ori= data
            

            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            thermal = Variable(thermal).cuda()
            labels = Variable(labels).cuda()

          
            
            outputs = net(inputs,thermal)

            visualize_prediction1(image_ori,outputs,labels)
            
            log = '[iter %d]'  % \
                     (i)
            print(log)
            
if __name__ == '__main__':
    main()
