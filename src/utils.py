import cv2
import os
import torch
import segmentation_refinement as refine
from src.u2net import normPRED
from torchvision import transforms, utils
from torchvision import transforms
from PIL import Image


def cascade(image, mask, model_folder="model_folder/", device='cuda:0'):
    image = image.permute(1, 2, 0).numpy()
    refiner = refine.Refiner(device=device, model_folder=model_folder)
    output = refiner.refine(image, mask, fast=True, L=300)
    return output


def save_mask_model(model, data, root_test=None, save=False):
    all_mask = []
    for n, X_batch in enumerate(data):
        Y_pred = model(X_batch.reshape(1, 3, 128, 128))[0][0].cpu().detach()
        all_mask.append(Y_pred)
        if save == True and root_test != None:
            for i in Y_pred:
                utils.save_image(i, f'mask_model/{os.listdir(root_test)[n]}')
    return all_mask


def get_result(model, images_test, images_test_init, N, tol_u2net=0.2, tol_cascade=252):
    result = []
    mask_u2net = []
    for n in range(N):
        mask_model = normPRED(save_mask_model(model, [images_test[n]])[0][0]).numpy()
        cur_mask_u2net = torch.from_numpy(mask_model).reshape(1, 128, 128)
        cur_mask_u2net = cur_mask_u2net>=tol_u2net
        mask_u2net.append(cur_mask_u2net)
        cur_result = torch.from_numpy(cascade(images_test_init[n], (mask_model>=tol_u2net)*255))
        cur_result = (cur_result>=tol_cascade)
        result.append(cur_result.reshape(1, 128, 128))
    return mask_u2net, result

