'''
Names (Please write names in <Last Name, First Name> format):
1. kanagaraj, kanimozhi
2. Patel, Jaimin
3. Murphy, Jamison

TODO: Project type
Image Segmentation
dataset : VOCSegmentation-2012

TODO: Report what each member did in this project

'''
import argparse
import torch
import torchvision
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

    def __init__(self,input_features, n_class, pool='max'):
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
        
        # convolution layers (5)
        #Layer 1
        self.conv1 = torch.nn.Conv2d(input_features, 64, kernel_size=(3, 3), stride=2, padding=1)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv1_2 = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=2, padding=1)
        self.relu1_2 = torch.nn.ReLU(inplace=True)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        
        #Layer 2
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.conv2_2 = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=2, padding=1)
        self.relu2_2 = torch.nn.ReLU(inplace=True)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        
        #Layer 3
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=2, padding=1)
        self.relu3 = torch.nn.ReLU(inplace=True)
        self.conv3_2 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=2, padding=1)
        self.relu3_2 = torch.nn.ReLU(inplace=True)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        
        #Layer 4
        self.conv4 = torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=2, padding=1)
        self.relu4 = torch.nn.ReLU(inplace=True)
        self.conv4_2 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=2, padding=1)
        self.relu4_2 = torch.nn.ReLU(inplace=True)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        
        #Layer 5
        self.conv5 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=2, padding=1)
        self.relu5 = torch.nn.ReLU(inplace=True)
        self.conv5_2 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=2, padding=1)
        self.relu5_2 = torch.nn.ReLU(inplace=True)
        self.pool5 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        
        #Layer 6
        self.conv6 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=2, padding=1)
        self.relu6 = torch.nn.ReLU(inplace=True)
        self.conv6_2 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=2, padding=1)
        self.relu6_2 = torch.nn.ReLU(inplace=True)
        self.pool6 = torch.nn.AvgPool2d(kernel_size=3, stride=2)
        
        #Layer 7
        self.conv7 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=2, padding=1)
        self.relu7 = torch.nn.ReLU(inplace=True)
        self.conv7_2 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=2, padding=1)
        self.relu7_2 = torch.nn.ReLU(inplace=True)
        self.pool7 = torch.nn.AvgPool2d(kernel_size=3, stride=2)
        
        #Layer 8
        self.conv8 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=2, padding=1)
        self.relu8 = torch.nn.ReLU(inplace=True)
        self.conv8_2 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=2, padding=1)
        self.relu8_2 = torch.nn.ReLU(inplace=True)
        self.pool8 = torch.nn.AvgPool2d(kernel_size=3, stride=2)
        
    
        #Fully convolution layers
        #fc 1
        self.fcn_1 = torch.nn.Conv2d(512, 4096, kernel_size=7)
        self.relu_fc_1 = torch.nn.ReLU(inplace=True)
        self.drop1 = torch.nn.Dropout2d()
        
        #fc 2
        self.fcn_2 = torch.nn.Conv2d(4096, 1, kernel_size=1)
        self.relu_fc_2 = torch.nn.ReLU(inplace=True)
        self.drop2 = torch.nn.Dropout2d()
        
        self.score  = torch.nn.Conv2d(4096, n_class, 1)
        self.score_pool3 = torch.nn.Conv2d(256, n_class, 1)
        self.score_pool4 = torch.nn.Conv2d(512, n_class, 1)
        
        #upsampling
        
        self.upsample_1= torch.nn.ConvTranspose2d(
            n_class, n_class, 8, stride=2, bias=False)
        self.upsampling_2 = torch.nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, bias=False)
        self.upsample_pool= torch.nn.ConvTranspose2d(
            n_class, n_class, 8, stride=2, bias=False)
        
    def forward(self, x):
        '''
            Args:
                x : torch.Tensor
                    tensor of N x d

            Returns:
                torch.Tensor
                    tensor of n_output
        '''

        # TODO: Implement forward function
        c = x
       
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv2_2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv3_2(x)
        x = self.pool3(x)
        pool3 = x
        
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv4_2(x)
        x = self.pool4(x)
        pool4 = x
        
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.conv5_2(x)
        x = self.pool5(x)
        
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.conv6_2(x)
        x = self.pool6(x)
        
        x = self.conv7(x)
        x = self.relu7(x)
        x = self.conv7_2(x)
        x = self.pool7(x)
        
        x = self.conv8(x)
        x = self.relu8(x)
        x = self.conv8_2(x)
        x = self.pool8(x)
        
        x = self.relu_fc_1(self.fcn_1(x))
        x = self.drop1(x)

        x = self.relu_fc_2(self.fcn_2(x))
        x = self.drop2(x)

        #yet to forward the transposed layer(upsampling)       
        
        return x

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
    loss_func = torch.nnCrossEntropyLoss()

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

            # TODO: Compute loss function
            loss = loss_func(outputs, labels)

            # TODO: Update parameters by backpropagation
            loss.backward()
            optimizer.step()

            # TODO: Accumulate total loss for the epoch
            total_loss = total_loss + loss.item()
      
        mean_loss = total_loss / float(batch)

        # Log average loss over the epoch
        print('Epoch=%d  Loss: %.3f' % (epoch + 1, mean_loss))

    return net

def evaluate(net, dataloader, classes):
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
     n_correct = 0
    n_sample = 0
    
    # Make sure we do not backpropagate
    with torch.no_grad():

        for (images, labels) in dataloader:
            
            
            shape = images.shape
            n_dim = np.prod(shape[1:])
            images = images.view(-1, n_dim)
            
            # TODO: Forward through the network
            outputs = net(images)
            
            # TODO: Compute evaluation metric(s) for each sample
            _, predictions = torch.max(outputs, dim=1)
            n_sample = n_sample + labels.shape[0]
            n_correct = n_correct + torch.sum(predictions == labels).item()
            
            

    # TODO: Compute mean evaluation metric(s)
    mean_accuracy = 100.0 * n_correct / n_sample
    Intersetion_union = intersection_over_union(predictions, images)
    
    # TODO: Print scores
    print('Jaccard Index of  %d images: %d %% ' % (n_sample, Intersetion_union))
    print('Mean accuracy over %d images: %d %%' % (n_sample, mean_accuracy))
    
    # TODO: Convert the last batch of images back to original shape
    images = images.view(shape[0], shape[1], shape[2], shape[3])
    images = images.cpu().numpy()
    images = np.transpose(images, (0, 2, 3, 1))

    # TODO: Convert the last batch of predictions to the original image shape
    predictions = predictions.view(shape[0], shape[1], shape[2], shape[3])
    predictions = predictions.cpu().numpy()
    predictions = np.transpose(predictions, (0, 2, 3, 1))
    
    # TODO: Plot images
    plot_images(
        X=images, 
        n_row=2, 
        n_col=2, 
        fig_title='Image Segmentation of fully convolutional neural networks uf VOC-2012 dataset')

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
    intersection = np.logical_and(ground_truth, prediction)
    union = np.logical_or(ground_truth, prediction)
    score_IOU = np.sum(intersection) / np.sum(union)
    
    return score_IOU

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

    fig = plt.figure()
    fig.suptitle(fig_title)

    for i in range(1, n_row * n_col + 1):

        ax = fig.add_subplot(n_row, n_col, i)

        index = i - 1
        if index >= (n_row * n_col /  2):
            x_i = Y[index, ...]
        else:
            x_i = X[index, ...]
        subplot_title_i = subplot_titles[index]

        if len(x_i.shape) == 1:
            x_i = np.expand_dims(x_i, axis=0)

        ax.set_title(subplot_title_i)
        ax.imshow(x_i)

        plt.box(False)
        plt.axis('off')
        
    plt.show()



if __name__ == '__main__':

    # TODO: Set up data preprocessing step
    # https://pytorch.org/docs/stable/torchvision/transforms.html
    data_preprocess_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224,224)),
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
        'tvmonitor'
    ]
    #Deining the input features
    input_features = 3 * 224 * 224
    
    #VOC has 20 classes
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
        weight_decay=args.weight_decay,
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
        dataloader=dataloader_test,
        classes=classes)
