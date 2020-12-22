'''
Names (Please write names in <Last Name, First Name> format):
1. Doe, John
2. Doe, Jane

TODO: Project type

TODO: Report what each member did in this project

#python fcn_iou.py --train_network --batch_size 10 --n_epoch 5 --learning_rate 1e-3 --lambda_weight_decay 0.01 --learning_rate_decay 0.80 --learning_rate_decay_period 2
'''
import argparse
import torch, torchvision
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()

# Commandline arguments
parser.add_argument('--train_network',
    action='store_true', help='If set, then trains network')
parser.add_argument('--batch_size',
    type=int, default=4, help='Number of samples per batch')
parser.add_argument('--n_epoch',
    type=int, default=1, help='Number of times to iterate through dataset')
parser.add_argument('--learning_rate',
    type=float, default=1e-8, help='Base learning rate (alpha)')
parser.add_argument('--learning_rate_decay',
    type=float, default=0.50, help='Decay rate for learning rate')
parser.add_argument('--learning_rate_decay_period',
    type=float, default=1, help='Period before decaying learning rate')
parser.add_argument('--momentum',
    type=float, default=0.90, help='Momentum discount rate (beta)')
parser.add_argument('--lambda_weight_decay',
    type=float, default=0.0, help='Lambda used for weight decay')

# TODO: please add additional necessary commandline arguments here


args = parser.parse_args()


class FullyConvolutionalNetwork(torch.nn.Module):
    '''
    Fully convolutional network

    Args:
        Please add any necessary arguments
    '''

    def __init__(self,input_features, n_class):
        super(FullyConvolutionalNetwork, self).__init__()

        # TODO: Design your neural network using
        # (1) convolutional layers
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # (2) max pool layers
        # https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.max_pool2d
        # (3) average pool layers
        # https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.avg_pool2d
        # (4) transposed convolutional layers
        # https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
        
        #Layer 1 + Relu activation function + maxpooling
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv1_2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.relu1_2 = torch.nn.ReLU(inplace=True)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        #Layer 2 + Relu activation function + maxpooling
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.conv2_2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.relu2_2 = torch.nn.ReLU(inplace=True)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        #Layer 3 + Relu activation function + Avgpooling
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = torch.nn.ReLU(inplace=True)
        self.conv3_2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.relu3_2 = torch.nn.ReLU(inplace=True)
        self.pool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        
        #Layer 4 + Relu activation function , Latent vector
        self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.relu4 = torch.nn.ReLU(inplace=True)
        self.conv4_2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.relu4_2 = torch.nn.ReLU(inplace=True)
        #self.pool4 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        
        #Tranposing Convolutional layer
        #Fully connected layer 1
        self.fc1 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2)
        self.conv5 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.relu5 = torch.nn.ReLU(inplace=True)
        self.conv5_2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.relu5_2 = torch.nn.ReLU(inplace=True)
        
        #fully connected Layer 2
        self.fc2 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2)
        self.conv6 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.relu6 = torch.nn.ReLU(inplace=True)
        self.conv6_2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.relu6_2 = torch.nn.ReLU(inplace=True)
        
        
        #fully connected Layer 3
        self.fc3 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2)
        self.conv7 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.relu7 = torch.nn.ReLU(inplace=True)
        self.conv7_2 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.relu7_2 = torch.nn.ReLU(inplace=True)
        
        #fully connected Layer 4
        self.fc4 = torch.nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2)
        self.conv8 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.relu8 = torch.nn.ReLU(inplace=True)
        self.conv8_2 = torch.nn.Conv2d(in_channels=32, out_channels=n_class, kernel_size=3, padding=1)
        self.relu8_2 = torch.nn.ReLU(inplace=True)
        
        self.upsampling = torch.nn.Upsample((input_features , input_features))


    def forward(self, x):
        f = x
        f = self.relu1(self.conv1(f))
        f = self.relu1_2(self.conv1_2(f))
        f = self.pool1(f)

        f = self.relu2(self.conv2(f))
        f = self.relu2_2(self.conv2_2(f))
        f = self.pool2(f)

        f = self.relu3(self.conv3(f))
        f = self.relu3_2(self.conv3_2(f))
        f = self.pool3(f)
      

        f = self.relu4(self.conv4(f))
        f = self.relu4_2(self.conv4_2(f))
        #f = self.pool4(f)
  
        
        #Deconvolution/upsampling
        #fully connected Layer 1
        f = self.fc1(f)
        f = self.relu5(self.conv5(f))
        f = self.relu5_2(self.conv5_2(f))
        #fully connected Layer 2
        f = self.fc2(f)
        f = self.relu6(self.conv6(f))
        f = self.relu6_2(self.conv6_2(f))
        #fully connected Layer 3
        f = self.fc3(f)
        f = self.relu7(self.conv7(f))
        f = self.relu7_2(self.conv7_2(f))
        #fully connected Layer 4
        f = self.fc4(f)
        f = self.relu8(self.conv8(f))
        f = self.relu8_2(self.conv8_2(f))
        
        f = self.upsampling(f)
        
        return f
    
def train(net,
          dataloader,
          n_epoch,
          optimizer,
          learning_rate_decay,
          learning_rate_decay_period):
    '''
    Trains the network using a learning rate scheduler

    Args:
        net : torch.nn.Module
            neural network
        dataloader : torch.utils.data.DataLoader
            # https://pytorch.org/docs/stable/data.html
            dataloader for training data
        n_epoch : int
            number of epochs to train
        optimizer : torch.optim
            https://pytorch.org/docs/stable/optim.html
            optimizer to use for updating weights
        learning_rate_decay : float
            rate of learning rate decay
        learning_rate_decay_period : int
            period to reduce learning rate based on decay e.g. every 2 epoch

        Please add any necessary arguments

    Returns:
        torch.nn.Module : trained network
    '''

    # TODO: Define loss function
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(n_epoch):

        # Accumulate total loss for each epoch
        total_loss = 0.0

        # TODO: Decrease learning rate when learning rate decay period is met
        # e.g. decrease learning rate by a factor of decay rate every 2 epoch
        if epoch and epoch % learning_rate_decay_period == 0:
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate_decay * param_group['lr']
                
        for batch, (images, labels) in enumerate(dataloader):
            
            
            # TODO: Forward through the network
            outputs = net(images)
            
            # TODO: Clear gradients so we don't accumlate them from previous batches
            optimizer.zero_grad()
            labels = torch.squeeze(labels, dim=1)
            labels = labels.long()
            
            # TODO: Compute loss function
            loss = loss_func(outputs, labels)

            # TODO: Update parameters by backpropagation
            loss.backward()
            optimizer.step()

            # TODO: Accumulate total loss for the epoch
            total_loss = total_loss + loss.item()
      
        mean_loss = (total_loss / float(batch))

        # Log average loss over the epoch
        print('Epoch=%d  Loss: %.3f' % (epoch + 1, mean_loss))

    return net

def evaluate(net, dataloader):
    '''
    Evaluates the network on a dataset

    Args:
        net : torch.nn.Module
            neural network
        dataloader : torch.utils.data.DataLoader
            # https://pytorch.org/docs/stable/data.html
            dataloader for training data

        Please add any necessary arguments
    '''
    n_sample = 0
    
    # Make sure we do not backpropagate
    with torch.no_grad():

        for (images, labels) in dataloader:
            
            shape = images.shape
      
            # TODO: Forward through the network
            outputs = net(images)
            
            
            # TODO: Compute evaluation metric(s) for each sample
            #Take the argmax over the outputs
            _, predictions = torch.max(outputs, dim=1)
            
            target = torch.squeeze(labels)
            target = target.long()
            
            #print('Predictions shape', predictions.shape)
            #print('Target shape', target.shape)
            
            # Accumulate number of samples
            n_sample = n_sample + labels.shape[0]
            #print("predictions shape:", predictions.shape)
            #print("labels shape:", labels.shape)
            
    # TODO: Compute mean evaluation metric(s)
    IOU = intersection_over_union(predictions, target)
    
    # TODO: Print scores
    print('Jaccard over {} images{:.2f}%:'.format(n_sample, (IOU * 100.0)))
    
    # TODO: Convert the last batch of images back to original shape
    images = images.view(shape[0], shape[1], shape[2], shape[3])
    images = images.cpu().numpy()
    images = np.transpose(images, (0, 2, 3, 1))
    
    # TODO: Convert the last batch of predictions to the original image shape
    '''predictions = predictions.view(shape[0], shape[1], shape[2], shape[3])
    predictions = predictions.cpu().numpy()
    predictions = np.transpose(predictions, (0, 2, 3, 1))'''
    
    # TODO: Plot images
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.suptitle('Image Evaluation')

    ax.plot(images, IOU, marker='o', color='b', label='MSE')
    ax.legend(loc='best')
    ax.set_xlabel('images')
    ax.set_ylabel('IOU')

    # Show plots
    plt.show()
  
    
def intersection_over_union(prediction, ground_truth):
    '''
    Computes the intersection over union (IOU) between prediction and ground truth

    Args:
        prediction : numpy
            N x h x w prediction
        ground_truth : numpy
            N x h x w ground truth

    Returns:
        float : intersection over union
    '''

    # TODO: Computes intersection over union score
    # Implement ONLY if you are working on semantic segmentation
    # J(A, B) = |A over B| / |A cup B| = |A over B| / |A| + |B| - |A over B|
    iou = 0.0
    for target in ground_truth:
        intersection = torch.sum((prediction==ground_truth) * (ground_truth==target))
       
        union = torch.sum((prediction==target)+(ground_truth==target))

        inter_union = (intersection / union)
        iou = iou + inter_union
        

    avg_iou = float(iou / len(ground_truth))

    
    return avg_iou  

    
   


def plot_images(X, Y, n_row, n_col, fig_title, subplot_titles):
    '''
    Creates n_row by n_col panel of images

    Args:
        X : numpy
            N x h x w input data
        Y : numpy
            N x h x w predictions
        n_row : int
            number of rows in figure
        n_col : list[str]
            number of columns in figure
        fig_title : str
            title of plot

        Please add any necessary arguments
    '''

    fig = plt.figure()
    fig.suptitle(fig_title)

    # TODO: Visualize your input images and predictions
    for i in range(1, n_row * n_col + 1):

        ax = fig.add_subplot(n_row, n_col, i)

        x_i = X[i, ...]

        if len(x_i.shape) == 1:
            x_i = np.expand_dims(x_i, axis=0)

        ax.set_title(subplot_titles[i])
        ax.imshow(x_i)

        plt.box(False)
        plt.axis('off')


if __name__ == '__main__':

    # TODO: Set up data preprocessing step
    # https://pytorch.org/docs/stable/torchvision/transforms.html
    data_preprocess_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32,32)),
        torchvision.transforms.ToTensor(),
    ])

    # Download and setup your training set
    dataset_train = torchvision.datasets.VOCSegmentation(
        root='./data',
        year='2012',
        image_set='train',
        download=True,
        transform=data_preprocess_transform,
        target_transform=data_preprocess_transform)
    
    # Setup a dataloader (iterator) to fetch from the training set
    dataloader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,)
    
    # Download and setup your validation/testing set
    dataset_test = torchvision.datasets.VOCSegmentation(
        root='./data',
        year='2012',
        image_set='val',
        download=True,
        transform=data_preprocess_transform,
        target_transform=data_preprocess_transform)

    # TODO: Setup a dataloader (iterator) to fetch from the validation/testing set
    dataloader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2)
    
    #Defining the classes
    classes = [
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'pottedplant',
        'sheep',
        'sofa',
        'train',
        'tvmonitor']
    #Deining the input features
    input_features = 32
    
    #VOC has 21 classes
    n_class = 20

    # TODO: Define network
    net = FullyConvolutionalNetwork(
        input_features=input_features,
        n_class=n_class)

    # TODO: Setup learning rate optimizer
    # https://pytorch.org/docs/stable/optim.html?#torch.optim.SGD
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=args.learning_rate,
        weight_decay=args.lambda_weight_decay,
        momentum=args.momentum)
    
    if args.train_network:
        # Set network to training mode
        net.train()

        # TODO: Train network and save into checkpoint
        net = train(
            net=net,
            dataloader=dataloader_train,
            n_epoch=args.n_epoch,
            optimizer=optimizer,
            learning_rate_decay=args.learning_rate_decay,
            learning_rate_decay_period=args.learning_rate_decay_period)

        # Saves weight to checkpoint
        torch.save({ 'state_dict' : net.state_dict()}, './checkpoint.pth')
    else:
        # Load network from checkpoint
        checkpoint = torch.load('./checkpoint.pth')
        net.load_state_dict(checkpoint['state_dict'])

    # Set network to evaluation mode
    net.eval()

    # TODO: Evaluate network on testing set
    evaluate(
       net=net,
       dataloader=dataloader_test)
