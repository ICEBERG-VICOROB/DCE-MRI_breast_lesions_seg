# Importing needed libraries
import os
import random
import torch
import numpy as np
from utils import MRI_DataPatchLoader
from utils import get_inference_patches, reconstruct_image
from torch.utils.data import DataLoader
from torch.optim import Adadelta
import nibabel as nib
import SimpleITK as sitk
import pandas as pd
from metrics import DSC_seg
import csv
import time as time

#####################################################################################################
seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

#####################################################################################################
# Define some functions
########################################
# function to save image array as nifti
def save_image(image_arr, image_org, output_file):
    # create a itk image
    image = sitk.GetImageFromArray(image_arr, isVector=False)
    # Fill basic
    image.SetSpacing(image_org.GetSpacing())
    image.SetOrigin(image_org.GetOrigin())
    image.SetDirection(image_org.GetDirection())
    # Write image
    sitk.WriteImage(image, output_file, True)

########################################
# function to save data in csv file
def save_to_csv(filepath, dict_list, append=False):
    """Saves a list of dictionaries as a .csv file.

    :param str filepath: the output filepath
    :param List[Dict] dict_list: The data to store as a list of dictionaries.
        Each dictionary will correspond to a row of the .csv file with a column for each key in the dictionaries.
    :param bool append: If True, it will append the contents to an existing file.

    :Example:

    save_to_csv('data.csv', [{'id': '0', 'score': 0.5}, {'id': '1', 'score': 0.8}])
    """
    assert isinstance(dict_list, list) and all([isinstance(d, dict) for d in dict_list])
    open_mode = 'a' if append else 'w+'
    with open(filepath, mode=open_mode) as f:
        csv_writer = csv.DictWriter(f, dict_list[0].keys(), restval='', extrasaction='raise', dialect='unix')
        if not append or os.path.getsize(filepath) == 0:
            csv_writer.writeheader()
        csv_writer.writerows(dict_list)

########################################
# loss functions:

# function for dice coefficient
def dice_coef(y_true, y_pred):
    smooth = 1.

    tflat = y_true.reshape(-1)
    pflat = y_pred.reshape(-1)
    intersection = (tflat * pflat).sum()

    return ((2. * intersection + smooth) /
            (tflat.sum() + pflat.sum() + smooth))

# function for dice coefficient for multi-class
def dice_coef_multilabel(y_true, y_pred, numLabels=4):
    dice = 0
    for index in range(numLabels):
        dice -= dice_coef(y_true[:, 0, :, :, :] == index, y_pred[:, index, :, :, :])
        dice_total = 4 + dice
    return dice_total

# function for dice coefficient for binary classification
def dice_coef_singlelabel(y_true, y_pred):
    index = 1
    dice = -dice_coef(y_true[:, 0, :, :, :] == index, y_pred[:, index, :, :, :])
    return dice

# function for combined cross-entropy and dice loss
def CE_Dice_loss(y_true, y_pred):
    CE_loss = F.cross_entropy(torch.log(torch.clamp(y_pred, 1E-7, 1.0)),
                                       y_true.squeeze(dim=1).long(), ignore_index=2)
    Dice_loss = dice_coef_singlelabel(y_true, y_pred)

    loss = CE_loss + Dice_loss

    return loss

'''def DICESEN_loss(y_true, y_pred):
    smooth = 0.00000001
    index = 1
    #y_true = np.where(y_true[:, 0, :, :, :] == index)
    y_pred = y_pred[:, index, :, :, :]
    tflat = y_true.reshape(-1)
    pflat = y_pred.reshape(-1)
    intersection = (tflat * pflat).sum()
    dice = (2. * intersection) / ((tflat * tflat).sum() + (pflat * pflat).sum() + smooth)
    sen = (1. * intersection) / ((tflat * tflat).sum() + smooth)
    return 2-dice-sen'''

# Function for Dice-sensitivity loss (by Zhang et al.)
# https://github.com/MaciejMazurowski/mri-breast-tumor-segmentation/blob/master/Models_3D.py
def DICESEN_loss(y_true, y_pred):
    smooth = 0.00000001
    #print(y_true.shape,y_pred.shape)
    #y_true_f = y_true[:, 0, :, :, :]
    #print(y_true_f.shape)
    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred[:, 1, :, :, :]
    #print(y_pred_f.shape)
    y_pred_f = y_pred_f.reshape(-1)
    #print(y_true_f.shape, y_pred_f.shape)
    intersection = torch.sum(torch.mul(y_true_f,y_pred_f))
    dice= (2. * intersection ) / (torch.mul(y_true_f,y_true_f).sum() + torch.mul(y_pred_f,y_pred_f).sum() + smooth)
    sen = (1. * intersection ) / (torch.mul(y_true_f,y_true_f).sum() + smooth)
    return 2-dice-sen

#####################################################################################################
# A class for Early stopping
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, checkpoint_out):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, checkpoint_out)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, checkpoint_out)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, checkpoint_out):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), checkpoint_out)
        self.val_loss_min = val_loss

#####################################################################################################
# Modified U-Net model
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F


class Conv3dAuto(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2,
                        self.kernel_size[2] // 3)  # dynamic add padding based on the kernel_size


conv3x3 = partial(Conv3dAuto, kernel_size=3, bias=False)


def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        # print("a:",residual)
        if self.should_apply_shortcut:
            residual = self.shortcut(x)
            # print("b:",residual)
        x = self.blocks(x)
        # print("c:",x)
        x += residual
        # print("d:",x)
        x = self.activate(x)
        # print("f:",x)
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(
            nn.Conv3d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            nn.InstanceNorm3d(self.expanded_channels)) if self.should_apply_shortcut else None

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs), nn.InstanceNorm3d(out_channels))


class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )


class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1

        self.blocks = nn.Sequential(
            block(in_channels, out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion,
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x

########################################
class Unet(nn.Module):
    """
    Basic U-net model
    """

    def __init__(self, input_size, output_size, p=0.0):
        super(Unet, self).__init__()

        self.drop_layer = nn.Dropout(p=p)

        # conv1 down
        self.conv1 = ResNetBasicBlock(input_size, 32)

        # max-pool 1
        self.pool1 = nn.Conv3d(in_channels=32,
                               out_channels=32,
                               kernel_size=2,
                               stride=2)
        # conv2 down
        self.conv2 = ResNetBasicBlock(32, 64)

        # max-pool 2
        self.pool2 = nn.Conv3d(in_channels=64,
                               out_channels=64,
                               kernel_size=2,
                               stride=2)
        # conv3 down
        self.conv3 = ResNetBasicBlock(64, 128)

        # max-pool 3
        self.pool3 = nn.Conv3d(in_channels=128,
                               out_channels=128,
                               kernel_size=2,
                               stride=2)

        # conv4 down
        self.conv4 = ResNetBasicBlock(128, 256)

        # max-pool 4
        self.pool4 = nn.Conv3d(in_channels=256,
                               out_channels=256,
                               kernel_size=2,
                               stride=2)

        # conv5 down (latent space)
        self.conv5x = ResNetBasicBlock(256, 512)

        # up-sample conv5
        self.up0 = nn.ConvTranspose3d(in_channels=512,
                                      out_channels=256,
                                      kernel_size=2,
                                      stride=2)

        self.conv5a = ResNetBasicBlock(256, 256)

        # up-sample conv4
        self.up1 = nn.ConvTranspose3d(in_channels=256,
                                      out_channels=128,
                                      kernel_size=2,
                                      stride=2)

        # conv 5 (add up1 + conv3)
        self.conv5 = ResNetBasicBlock(128, 128)

        # up-sample conv5
        self.up2 = nn.ConvTranspose3d(in_channels=128,
                                      out_channels=64,
                                      kernel_size=2,
                                      stride=2)

        # conv6 (add up2 + conv2)
        self.conv6 = ResNetBasicBlock(64, 64)

        # up 3
        self.up3 = nn.ConvTranspose3d(in_channels=64,
                                      out_channels=32,
                                      kernel_size=2,
                                      stride=2)

        # conv7 (add up3 + conv1)
        self.conv7 = ResNetBasicBlock(32, 32)

        # conv8 (classification)
        self.conv8 = nn.Conv3d(in_channels=32,
                               out_channels=output_size,
                               kernel_size=1)

    def forward(self, x):
        # encoder
        x1 = self.drop_layer(F.relu(self.conv1(x)))

        x1p = self.pool1(x1)

        x2 = self.drop_layer(F.relu(self.conv2(x1p)))

        x2p = self.pool2(x2)

        x3 = self.drop_layer(F.relu(self.conv3(x2p)))

        x3p = self.pool3(x3)

        x4 = self.drop_layer(F.relu(self.conv4(x3p)))

        x4p = self.pool4(x4)

        # latent space
        x5a = self.drop_layer(F.relu(self.conv5x(x4p)))

        # decoder
        up0 = self.up0(x5a)

        x5b = self.drop_layer(F.relu(self.conv5a(up0 + x4)))

        up1 = self.up1(x5b)

        x5 = self.drop_layer(F.relu(self.conv5(up1 + x3)))

        up2 = self.up2(x5)

        x6 = self.drop_layer(F.relu(self.conv6(up2 + x2)))

        up3 = self.up3(x6)

        x7 = self.drop_layer(F.relu(self.conv7(up3 + x1)))

        # output layer (2 classes)
        # we use a softmax layer to return probabilities for each class
        out = F.softmax(self.conv8(x7), dim=1)
        return out
#####################################################################################################
#####################################################################################################
# Data Preparation

# data path
data_path = "/home/rua/dataset/BH_subset"

# train/validation split percentage
train_split = 0.2

# input modality names
#(ex: pre-contrast, last post-contrast, standard deviation)
input_data = ['pre_raw.nii.gz', 'post_last_raw.nii.gz', 'std.nii.gz']

# ground-truth name
gt_data = 'modified_GT/gt_raw.nii.gz'

# ROI name
roi_data = 'ROI_2.nii.gz'

#output folder name
out_name = 'trial_0000'

# additional options for patch size, sampling step, normalization, etc...
patch_size = (32, 32, 32)
sampling_step = (16, 16, 16)
normalize = True
batch_size = 32
sampling_type = 'hybrid'

# some training options
gpu_use = True
num_epochs = 20
num_folds = 5
num_pos_samples = 2000
patience_es = 15
#####################################################################################################
# to write the experiment info in txt file

# first save them in a dictionary
options = {}

# data path
options['data_path'] = data_path

# train/validation split percentage
options['train_split'] = train_split

# input modality names
options['input_data'] = input_data

# ground-truth name
options['gt_data'] = gt_data

# ROI name
options['roi_data'] = roi_data

#output folder name
options['out_name'] = out_name

# additional options for patch size, sampling step, normalization, etc...
options['patch_size'] = patch_size
options['sampling_step'] = sampling_step
options['normalize'] = normalize
options['batch_size'] = batch_size
options['sampling_type'] = sampling_type

# some training options
options['gpu_use'] = gpu_use
options['num_epochs'] = num_epochs
options['num_folds'] = num_folds
options['num_pos_samples'] = num_pos_samples
options['patience_es'] = patience_es


# uncomment the following to save experiment info in txt file
'''report = os.path.join('/home/rua/codes/modified/models', out_name, 'report.txt')
f = open(report, 'a')
f.write('data and training info: \n' + repr(options) + '\n')
f.close()'''
#####################################################################################################
# get data paths and specify number of cases per fold
dataset_paths = [f.path for f in os.scandir(data_path) if f.is_dir()]
num_cases_per_fold = len(dataset_paths) // num_folds

#####################################################################################################
# training the model

all_metrics = []
for fold in range(num_folds):
    print("*********************************************************************************************")
    print("fold number {:d}".format(fold + 1))
    #write in txt file
    '''f = open(report, 'a')
    f.write('********************************************************************************************* \n'
            "fold number {:d}".format(fold + 1) + '\n')
    f.close()'''
    ####
    s = len(dataset_paths)//num_folds
    r = s * fold
    testing_paths = dataset_paths[r:r + s]
    training_paths = [x for j, x in enumerate(dataset_paths) if j not in range(r, r + s)]
    if fold == 4:
        left = len(dataset_paths) % num_folds
        testing_paths = dataset_paths[r:r + s + left]
        training_paths = [x for j, x in enumerate(dataset_paths) if j not in range(r, r + s + left)]

    # randomly shuffle the training dataset and divide it into training and validation sets
    random.shuffle(training_paths)

    # load training / validation data
    t_d = int(len(training_paths) * (1 - train_split))
    training_data = training_paths[:t_d]
    validation_data = training_paths[t_d:]

    input_train_data = {scan: [os.path.join(scan, d)
                               for d in input_data]
                        for scan in training_data}

    input_train_labels = {scan: [os.path.join(scan, gt_data)]
                          for scan in training_data}

    input_train_rois = {scan: [os.path.join(scan, roi_data)]
                        for scan in training_data}

    input_val_data = {scan: [os.path.join(scan, d)
                             for d in input_data]
                      for scan in validation_data}

    input_val_labels = {scan: [os.path.join(scan, gt_data)]
                        for scan in validation_data}

    input_val_rois = {scan: [os.path.join(scan, roi_data)]
                      for scan in validation_data}

    print("********************************")
    print('Training data: ')
    # write in txt file
    '''f = open(report, 'a')
    f.write('******************************** \n'
            'Training data: \n')
    f.close()'''
    ####
    training_dataset = MRI_DataPatchLoader(input_data=input_train_data,
                                           labels=input_train_labels,
                                           rois=input_train_rois,
                                           patch_size=patch_size,
                                           sampling_step=sampling_step,
                                           sampling_type=sampling_type,
                                           normalize=normalize,
                                           num_pos_samples=num_pos_samples)

    training_dataloader = DataLoader(training_dataset,
                                     batch_size=batch_size,
                                     shuffle=True)

    print("********************************")
    print('Validation data: ')
    # write in txt file
    '''f = open(report, 'a')
    f.write('******************************** \n'
            'Validation data: \n')
    f.close()'''
    ####
    validation_dataset = MRI_DataPatchLoader(input_data=input_val_data,
                                             labels=input_val_labels,
                                             rois=input_val_rois,
                                             patch_size=patch_size,
                                             sampling_step=sampling_step,
                                             sampling_type=sampling_type,
                                             normalize=normalize,
                                             num_pos_samples=num_pos_samples)

    validation_dataloader = DataLoader(validation_dataset,
                                       batch_size=batch_size,
                                       shuffle=True)

    print("********************************")
    # write in txt file
    '''f = open(report, 'a')
    f.write('******************************** \n')
    f.close()'''
    ####

    # uncomment the following line to initialize the early_stopping object
    '''early_stopping = EarlyStopping(patience=patience_es, verbose=True)'''

    # Define the U-net model
    # change input_size according to number of input scans
    lesion_model = Unet(input_size=3, output_size=2)

    # define the torch.device
    device = torch.device('cuda') if gpu_use else torch.device('cpu')

    # define the optimizer
    optimizer = Adadelta(lesion_model.parameters())

    # send the model to the device
    lesion_model = lesion_model.to(device)

    t0 = time.time()
    # training loop
    training = True
    epoch = 1
    try:
        while training:

            # epoch specific metrics
            train_loss = 0
            train_accuracy = 0
            val_loss = 0
            val_accuracy = 0

            # -----------------------------
            # training samples
            # -----------------------------

            # set the model into train mode
            lesion_model.train()
            for a, batch in enumerate(training_dataloader):
                # process batches: each batch is composed by training (x) and labels (y)
                # x = [32, 2, 32, 32, 32]
                # y = [32, 1, 32, 32, 32]

                x = batch[0].to(device)
                y = batch[1].to(device)

                # clear gradients
                optimizer.zero_grad()

                # infer the current batch
                pred = lesion_model(x)

                # compute the loss.
                # we ignore the index=2
                loss = F.cross_entropy(torch.log(torch.clamp(pred, 1E-7, 1.0)),
                                       y.squeeze(dim=1).long(), ignore_index=2)

                '''loss = dice_coef_singlelabel(y, pred)'''

                '''loss = DICESEN_loss(y, pred)'''
                train_loss += loss.item()

                # backward loss and next step
                loss.backward()
                optimizer.step()

                # compute the accuracy
                pred = pred.max(1, keepdim=True)[1]
                batch_accuracy = pred.eq(y.view_as(pred).long())
                train_accuracy += (batch_accuracy.sum().item() / np.prod(y.shape))

            # -----------------------------
            # validation samples
            # -----------------------------

            # set the model into train mode
            lesion_model.eval()
            for b, batch in enumerate(validation_dataloader):
                x = batch[0].to(device)
                y = batch[1].to(device)

                # infer the current batch
                with torch.no_grad():
                    pred = lesion_model(x)

                    # compute the loss.
                    # we ignore the index=2
                    loss = F.cross_entropy(torch.log(torch.clamp(pred, 1E-7, 1.0)),
                                           y.squeeze(dim=1).long(), ignore_index=2)

                    '''loss = dice_coef_singlelabel(y, pred)'''

                    '''loss = DICESEN_loss(y, pred)'''
                    val_loss += loss.item()

                    # compute the accuracy
                    pred = pred.max(1, keepdim=True)[1]
                    batch_accuracy = pred.eq(y.view_as(pred).long())
                    val_accuracy += batch_accuracy.sum().item() / np.prod(y.shape)

            # compute mean metrics
            train_loss /= (a + 1)
            train_accuracy /= (a + 1)
            val_loss /= (b + 1)
            val_accuracy /= (b + 1)

            # uncomment to use early stop
            '''checkpoint_out = os.path.join('models', out_name, 'checkpoint_' + 'fold' + str(fold + 1) + '_model' + str(epoch) + '.pt')
            early_stopping(val_loss, lesion_model, checkpoint_out)
            if early_stopping.early_stop:
                print("Early stopping")
                break'''

            # print epoch number and losses
            print('Epoch {:d} train_loss {:.4f} train_acc {:.4f} val_loss {:.4f} val_acc {:.4f}'.format(
                epoch,
                train_loss,
                train_accuracy,
                val_loss,
                val_accuracy))
            # uncomment to write epoch number and losses in txt file
            '''f = open(report, 'a')
            f.write('Epoch {:d} train_loss {:.4f} train_acc {:.4f} val_loss {:.4f} val_acc {:.4f}'.format(
                epoch,
                train_loss,
                train_accuracy,
                val_loss,
                val_accuracy) + '\n')
            f.close()'''
            ####

            # save weights
            # uncomment to save all models
            '''torch.save(lesion_model.state_dict(),
                       os.path.join('models', out_name, 'fold' + str(fold + 1) + '_model' + str(epoch) + '.pt'))'''
            # to save the last model only
            if epoch == 20:
                torch.save({
                    'fold': fold,
                    'epoch': epoch,
                    'model_state_dict': lesion_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, os.path.join('/home/rua/codes/modified/models', out_name, 'fold' + str(fold + 1) + '_model' + str(epoch) + '.pt'))

            t1 = time.time()
            print('Total time: ', round((t1 - t0) / 60, 1), 'minutes.\n')

            # update epochs
            epoch += 1

            if epoch > num_epochs:
                training = False
    except KeyboardInterrupt:
        pass

    t1 = time.time()
    print('Total time: ', round((t1 - t0) / 60, 1), 'minutes.\n')

    print("********************************")

    # uncomment to save total time in txt file
    '''f = open(report, 'a')
    f.write('Total time: '+ str(round((t1 - t0) / 60, 1)) + 'minutes.\n'
            '******************************** \n')
    f.close()'''

    #####################################################################################################
    # testing

    # obtain a list of test scans
    test_scans = testing_paths
    th = 0.5

    # iterate through the scans and evaluate the results
    metrics = np.zeros((len(test_scans), 1))
    csv_path = os.path.join('/home/rua/codes/modified/models', out_name, 'dices.csv')

    for i, scan_name in enumerate(test_scans):
        # print("scan name:", scan_name)
        scan_path = os.path.join(test_scans[i], 'post_last_raw.nii.gz')
        scan = nib.load(scan_path)

        infer_patches, coordenates = get_inference_patches(scan_path=test_scans[i],
                                                           input_data=input_data,
                                                           roi=roi_data,
                                                           patch_shape=patch_size,
                                                           step=sampling_step,
                                                           normalize=normalize)

        lesion_out = np.zeros((infer_patches.shape[0], 2, infer_patches.shape[2],infer_patches.shape[3],infer_patches.shape[4])).astype('float32')
        batch_size = batch_size

        # model testing
        lesion_model.eval()
        with torch.no_grad():
            for b in range(0, len(lesion_out), batch_size):
                x = torch.tensor(infer_patches[b:b + batch_size]).to(device)
                pred = lesion_model(x)
                lesion_out[b:b + batch_size] = pred.cpu().numpy()

        # reconstruct image
        lesion_prob = reconstruct_image(lesion_out[:, 1],
                                        coordenates,
                                        scan.shape)

        # save the probability image
        out = os.path.join('/home/rua/codes/modified/predictions', out_name)
        s = scan_name[-12:]
        print(s)

        output_file2 = os.path.join(out, 'probabilities', str(s) + "_prob.nii.gz")
        new2 = nib.Nifti1Image(lesion_prob, scan.affine, scan.header)
        nib.save(new2, output_file2)

        # binarize the results
        lesion_prob = (lesion_prob > th).astype('uint8')

        # save binary segmentation
        output_file = os.path.join(out, str(s) + "_pred.nii.gz")
        new = nib.Nifti1Image(lesion_prob, scan.affine, scan.header)
        nib.save(new, output_file)

        # evaluate and print dice
        gt = nib.load(os.path.join(test_scans[i], gt_data))
        dsc_metric = DSC_seg(gt.get_fdata() == 1, lesion_prob > 0)
        metrics[i] = [dsc_metric]
        all_metrics.append(metrics[i])

        print('SCAN:', scan_name, 'Dice: ', dsc_metric)
        # write in txt file
        '''f = open(report, 'a')
        f.write('SCAN:' + str(scan_name) + 'Dice: ' + str(dsc_metric) + '\n')
        f.close()'''
        ##########
        # save dices to csv file
        if (fold == 0) and (i == 0):
            save_to_csv(csv_path, [{'scan id': s, 'dice': dsc_metric}], append=False)
        else:
            save_to_csv(csv_path, [{'scan id': s, 'dice': dsc_metric}], append=True)


    # we use PANDAS to describe data :)
    m = pd.DataFrame(metrics, columns=['DSC'])
    print("********************************")
    print(m.describe().T)


m2 = pd.DataFrame(all_metrics, columns=['DSC'])
print("********************************")
print("********************************")
print(m2.describe().T)

#####################################################################################################

