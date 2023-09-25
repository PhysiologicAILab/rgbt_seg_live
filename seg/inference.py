import torch
from models.LSNet_materials import LSNet
from torch.backends import cudnn
import torchvision.transforms as transforms

cudnn.benchmark = True

class seg_inference(object):
    def __init__(self, ckpt_path) -> None:

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = LSNet()
        self.model = self.model.to(self.device).eval()
        self.model.load_state_dict(torch.load(ckpt_path))
        
        self.rgb_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.thermal_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()
            ])


    def run_inference(self, rgb_img, thermal_img=None):
        
        rgb_img_tensor = self.rgb_transform(rgb_img)
        # thermal_img_tensor = self.thermal_transform(thermal_img)

        pred_seg = self.model(rgb_img_tensor)
        # pred_seg = self.model(rgb_img, thermal_img)

        pred_seg = torch.argmax(pred_seg, 1)
        pred_seg = pred_seg.cpu().numpy()

        return pred_seg


