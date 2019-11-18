
from fastai.vision import *
import numpy as np


class MyLoss(nn.Module):
    def forward(self, yhat, bbox_tgts, class_tgts):
        det_loss=nn.L1Loss()(yhat[:,:4].unsqueeze_(dim=1), bbox_tgts)
        cls_loss=nn.CrossEntropyLoss()(yhat[:,4:], class_tgts.view(-1))
        #print(det_loss, cls_loss)
        
        return det_loss + 1.0*cls_loss


class Model(object):
    def __init__(self, path='/content/drive/My Drive/Colab Notebooks/bbc_train/', file='export.pkl'):
        
        self.learn=load_learner(path=path, file=file) #Load model
        self.class_names=['brick', 'ball', 'cylinder'] #Be careful here, labeled data uses this order, but fastai will use alphabetical by default!

    def predict(self, x):
        '''
        Input: x = block of input images, stored as Torch.Tensor of dimension (batch_sizex3xHxW), 
                   scaled between 0 and 1. 
        Returns: a tuple containing: 
            1. The final class predictions for each image (brick, ball, or cylinder) as a list of strings.
            2. Upper left and lower right bounding box coordinates (in pixels) for the brick ball 
            or cylinder in each image, as a 2d numpy array of dimension batch_size x 4.
            3. Segmentation mask for the image, as a 3d numpy array of dimension (batch_sizexHxW). Each value 
            in each segmentation mask should be either 0, 1, 2, or 3. Where 0=background, 1=brick, 
            2=ball, 3=cylinder. 
        '''
            
        #Normalize input data using the same mean and std used in training:
        #x_norm=normalize(x, torch.tensor(self.learn.data.stats[0]), torch.tensor(self.learn.data.stats[1]))
        #print(x_norm.shape)
        
        def compute_corner_locations(y, im_shape=(256,256)):
            shape_vec=np.array(im_shape*2)
            bounds=((y+1)*shape_vec/2).ravel()
            corners=np.array([bounds[0], bounds[1],bounds[2],bounds[3]])
            return corners
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            yhat=self.learn.model(x.to(device))
            yhat=yhat.detach().cpu()
        
        
        ybox=yhat[:,4:]
        class_predictions_ind= ybox.argmax(1)
        class_predictions=[self.learn.data.classes[i] for i in class_predictions_ind]
        #print(class_predictions)

        bbox=yhat[:,:4]
        bbox=[compute_corner_locations(bbox[i].cpu().numpy()) for i in range(yhat.shape[0])]
        #print(bbox)
 
        mask=np.random.randint(low=0, high=1, size=(x.shape[0], x.shape[2], x.shape[3]))
        
        '''
        class_predictions=[self.class_names[np.random.randint(3)] for i in range(x.shape[0])]
        bbox=np.random.rand(x.shape[0], 4)
        bbox[:,0] *= x.shape[2]; bbox[:,2] *= x.shape[2] 
        bbox[:,1] *= x.shape[3]; bbox[:,3] *= x.shape[3]       
        mask=np.random.randint(low=0, high=4, size=(x.shape[0], x.shape[2], x.shape[3]))
        '''
        return (class_predictions, bbox, mask)
