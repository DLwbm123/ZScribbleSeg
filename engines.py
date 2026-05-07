import math
import sys
import random
import time
import datetime
from typing import Iterable
from estimator import *
import torch.nn.functional as Func
import numpy as np
import torch
import torch.nn as nn
import util.misc as utils
from torch.autograd import Variable
from mixup import mixup_process, get_lambda
from torch.nn import functional as F
import torchvision
import matplotlib.pyplot as plt
from cutout import Cutout, rotate_invariant, rotate_back
from inference import keep_largest_connected_components
from spatial_function import *

class Visualize_train(nn.Module):
    def __init__(self):
        super().__init__()
        
    def save_image(self, image, tag, epoch, writer):
        image = (image - image.min()) / (image.max() - image.min() + 1e-6)
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)
        
    def forward(self, originals, puzzles, inputs, outputs, ori_labels, labels, puzzle_labels, mixed_labels,ori_outputs,outputs_puzzle, epoch, writer):
        self.save_image(originals, 'inputs_original', epoch, writer)
        self.save_image(puzzles, 'inputs_puzzles', epoch, writer)
        self.save_image(inputs, 'inputs_cut', epoch, writer)
        self.save_image(ori_labels.float(), 'labels_original', epoch, writer)
        self.save_image(puzzle_labels.float(), 'labels_puzzle', epoch, writer)
        self.save_image(labels.float(), 'labels_cut', epoch, writer)
        self.save_image(mixed_labels.float(), 'mixed_output', epoch, writer)
        self.save_image(outputs.float(), 'outputs_mixed', epoch, writer)
        self.save_image(ori_outputs.float(), 'ori_outputs', epoch, writer)
        self.save_image(outputs_puzzle.float(), 'outputs_puzzle', epoch, writer)

def convert_targets(targets, device):
    masks = [t["masks"] for t in targets]
    target_masks = torch.stack(masks)
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 5, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    return target_masks

def to_onehot(target_masks, device):
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 5, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    return target_masks


def to_onehot_dim4(target_masks, device):
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 4, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    return target_masks

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    dataloader_dict: dict, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, args, writer,estimate_alpha):

    model.train()
    criterion.train()
    # training profiling
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    total_train_time = 0.0
    train_steps = 0
    peak_mem_mb = 0.0
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 40
    numbers = { k : len(v) for k, v in dataloader_dict.items() }
    iterats = { k : iter(v) for k, v in dataloader_dict.items() }
    tasks = dataloader_dict.keys()
    counts = { k : 0 for k in tasks }
    total_steps = sum(numbers.values())
    start_time = time.time()
    original_list,puzzle_list, sample_list, output_list,output_original_list, target_list, target_ori_list,target_wo_cutout_list, output_mixed_list, output_puzzle_list =[],[],[], [], [], [], [], [], [],[]
    for step in range(total_steps):
        start = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        step_start_time = time.time()
        task = "MR"
        samples, targets = next(iterats[task])
        counts.update({task : counts[task] + 1 })
        datatime = time.time() - start
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if not isinstance(v, str)} for t in targets]
        targets_onehot= convert_targets(targets,device)

        model.eval()
        outputs = model(samples.tensors, task)
        with torch.no_grad():
            if estimate_alpha == True:
                outputs = model(samples.tensors, task)
                with torch.no_grad():
                    alpha_dict = {}
                    step_dict = {}
                    target_values = np.unique(np.argmax(targets_onehot.detach().cpu().numpy(),axis=1))
                    target_values = [int(i) for i in target_values if int(i) != 4]
                    omitted_values = [i for i in range(4) if i not in target_values]
                    for class_label in target_values:
                        n_probs = []
                        p_probs = []
                        g_probs = []
                        for batch_idx in range(targets_onehot.shape[0]):

                            u_index = 1-targets_onehot[batch_idx,4,:,:]
                            pos_index = targets_onehot[batch_idx,class_label,:,:]
                            p_prob = outputs["pred_masks"][batch_idx,class_label,:,:][u_index==1]
                            n_prob = outputs["pred_masks"][batch_idx,class_label,:,:][u_index==0]
                            g_prob = pos_index[u_index == 1]

                            n_probs = np.concatenate((n_probs, n_prob.cpu().detach().numpy()), axis=0)
                            p_probs = np.concatenate((p_probs, p_prob.cpu().detach().numpy()), axis=0)
                            g_probs = np.concatenate((g_probs, g_prob.cpu().detach().numpy()), axis=0)

                        n_probs = np.asarray(n_probs)
                        g_probs = np.asarray(g_probs)

                        step_dict.update({class_label:{"n":n_probs,"p":p_probs,"g":g_probs}})

                        n_probs = np.asarray(n_probs)
                        p_probs = np.asarray(p_probs)
                        g_probs = np.asarray(g_probs)

                        step_dict.update({class_label:{"n":n_probs,"p":p_probs,"g":g_probs}})

                    for class_label in target_values:
                        p_value = step_dict[class_label]["p"].reshape((len(step_dict[class_label]["p"]),-1))
                        n_value = step_dict[class_label]["n"].reshape((len(step_dict[class_label]["n"]),-1))
                        g_value = step_dict[class_label]["g"].reshape((len(step_dict[class_label]["g"]),-1))
                        alpha_dict.update({class_label:{"n":n_value,"p":p_value,"g":g_value}})

                    ratio_dict = EM_estimate(alpha_dict)

                    for value in omitted_values:
                        ratio_dict.update({value:0})

        model.train()
        # puzzlemix
        samples_var = Variable(samples.tensors, requires_grad=True)
        # puzzlemix -- parameters
        adv_p = 0.1
        adv_eps = 10.0

        adv_mask1 = np.random.binomial(n=1, p=adv_p)
        adv_mask2 = np.random.binomial(n=1, p=adv_p)

        noise=None
        if (adv_mask1 == 1 or adv_mask2 == 1):
            noise = torch.zeros_like(samples_var).uniform_(adv_eps/255., adv_eps/255.)
            input_noise = samples_var + noise
            samples_var = Variable(input_noise, requires_grad=True)

        ###
        outputs = model(samples_var, task)
        loss_dict = criterion(outputs, targets_onehot)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in ['loss_CrossEntropy'] if k in weight_dict)
        losses.backward(retain_graph=True)

        ### original output:
        output_original = model(samples_var,task)

        # puzzlemix -- unary
        unary = torch.sqrt(torch.mean(samples_var.grad **2, dim=1))
        unary = F.pad(unary, (22,22,22,22,0,0), 'constant')

        # calculate adversarial noise
        if (adv_mask1 == 1 or adv_mask2 == 1):
            noise += (adv_eps + 2) / 255. * samples_var.grad.sign()
            noise = torch.clamp(noise, -args.adv_eps/255., args.adv_eps/255.)
            adv_mix_coef = np.random.uniform(0,1)
            noise = adv_mix_coef * noise

        samples_var_256 = F.pad(samples_var, (22,22,22,22,0,0,0,0), 'constant')
        targets_onehot_256 = F.pad(targets_onehot, (22,22,22,22,0,0,0,0), 'constant')
        out, reweighted_target, indices_transport, mask_transport = mixup_process(samples_var_256, targets_onehot_256, args=args, grad= unary, noise = noise)
        out = out[:,:,22:-22,22:-22]
        reweighted_target = reweighted_target[:,:,22:-22,22:-22]
        mask_transport = mask_transport[:,:,22:-22,22:-22]
        torch.cuda.empty_cache()
        ###

        ##Cutout
        samples_cut, targets_cut, masks_cut = Cutout(out, reweighted_target, device)
        samples_cut, targets_cut, angles = rotate_invariant(samples_cut, targets_cut)
        masks_cut = masks_cut.to(device)
        targets_cut = targets_cut[:,0:4,:,:]
        outputs_cut = model(samples_cut, task)

        samples_cut_back, outputs_cut,targets_cut = rotate_back(samples_cut, outputs_cut["pred_masks"],targets_cut,angles)

        # original_loss
        loss_dict_ori = criterion(output_original, targets_onehot)
        weight_dict = criterion.weight_dict
        losses_ori = sum(loss_dict_ori[k] * weight_dict[k] for k in ['loss_CrossEntropy'] if k in weight_dict)
        losses_ori = losses_ori
        if step == 0:
            print("original loss:", losses_ori.item())

        # annotated area
        annotated_area = 1-targets_onehot[:,4:5,:,:]
        annotated_area = annotated_area.repeat(1,4,1,1)

        loss_gatedcrf_kernels_desc = [{"weight": 1, "xy": 6, "rgb": 0.1}]
        # adjust1 :5 -> 8
        loss_gatedcrf_radius = 8
        torch.cuda.empty_cache()

        if estimate_alpha == True:
            spatial_weight = ModelWeightGatedCRF()
            sample_for_crf = samples.tensors
            if isinstance(sample_for_crf, torch.Tensor) and sample_for_crf.device != device:
                sample_for_crf = sample_for_crf.to(device)
            spatial_prob = spatial_weight(output_original["pred_masks"],
                    loss_gatedcrf_kernels_desc,
                    loss_gatedcrf_radius,
                    sample_for_crf,
                    212,
                    212,
                )

        # estimate threshold
        if estimate_alpha == True:
            pseudo_labels = torch.zeros_like(output_original["pred_masks"])
            for key, value in ratio_dict.items():
                flat_labels = spatial_prob[:,key,:,:][targets_onehot[:,-1,:,:] == 1]
                sorted_dices = np.argsort(flat_labels.cpu().detach().numpy())
                sorted_labels = flat_labels[sorted_dices]
                try:
                    threshold_pseudo = sorted_labels[max(int(len(sorted_dices)*(1-ratio_dict[key]))-1,0)]
                    if step == 0:
                        print(key, ":", threshold_pseudo.item())
                    pseudo_labels[:,key,:,:][spatial_prob[:,key,:,:] < threshold_pseudo] = 1
                except:
                    pass

        annotated_area = 1-targets_onehot[:,4:5,:,:]
        annotated_area = annotated_area.repeat(1,4,1,1)

        if estimate_alpha == True:
            pseudo_labels = pseudo_labels*(1-annotated_area)
            label_pseudo = pseudo_labels.sum(1)
            label_pseudo[label_pseudo >= 1] = 1
            outputs_pseudo = output_original["pred_masks"]*(1-pseudo_labels)

            pseudo_loss = -label_pseudo * torch.log(outputs_pseudo.sum(1)+1e-12)
            pseudo_loss = pseudo_loss.mean()
            if step == 0:
                print("pseudo cut loss:", pseudo_loss.item())

        annotated_cut = targets_cut.sum(1, keepdim = True)

        # cutout_loss
        # gd_cut
        gd_loss = - targets_cut * torch.log(outputs_cut["pred_masks"]+1e-12)
        gd_loss = gd_loss.sum(1,keepdim=True) * annotated_cut
        gd_loss = gd_loss.mean()

        # integrity
        original_masks = output_original["pred_masks"]
        predictions_original_list = []

        for i in range(original_masks.shape[0]):
            prediction = np.uint8(np.argmax(original_masks[i,:,:,:].detach().cpu(), axis=0))
            prediction = keep_largest_connected_components(prediction)
            prediction = torch.from_numpy(prediction).to(device)
            predictions_original_list.append(prediction)

        predictions = torch.stack(predictions_original_list)
        predictions = torch.unsqueeze(predictions, 1)
        prediction_onehot = to_onehot_dim4(predictions,device)
        loss_dict_integ = criterion(output_original, prediction_onehot*(1-annotated_area))
        weight_dict = criterion.weight_dict
        losses_integ = sum(loss_dict_integ[k] * weight_dict[k] for k in ['loss_CrossEntropy'] if k in weight_dict)

        if step == 0:
            print("integrity loss:", losses_integ.item())

        ### mixed output
        mixed_output = torch.zeros_like(outputs_cut["pred_masks"])
        shuffled_output = output_original["pred_masks"][indices_transport].clone()
        for i in range(shuffled_output.shape[1]):
            mixed_output[:,i,:,:] = output_original["pred_masks"][:,i,:,:] * mask_transport[:,0,:,:] + shuffled_output[:,i,:,:] * (1-mask_transport[:,0,:,:])
        puzzle_output = mixed_output
        mixed_output = mixed_output*masks_cut

        if step % 20 == 0:
            for i in range(samples_var.shape[0]):
                original_list.append(samples_var[i])
                puzzle_list.append(out[i])
                target_wo_cutout_list.append(reweighted_target.argmax(1,keepdim=True)[i])
                sample_list.append(samples_cut_back[i])
                _, pre_masks = torch.max(outputs_cut['pred_masks'][i], 0, keepdims=True)
                output_list.append(pre_masks)
                _, pre_original = torch.max(output_original['pred_masks'][i], 0, keepdims=True)
                output_original_list.append(pre_original)
                target_ori_list.append(targets_onehot.argmax(1,keepdim=True)[i])
                target_list.append(targets_cut.argmax(1,keepdim=True)[i])
                output_mixed_list.append(mixed_output.argmax(1,keepdim=True)[i])
                output_puzzle_list.append(puzzle_output.argmax(1,keepdim=True)[i])

        # invariant_loss
        invariant_loss = 1- Func.cosine_similarity(outputs_cut["pred_masks"], mixed_output, dim=1).mean()
        invariant_loss = 0.05*invariant_loss
        if step == 0:
            print("invariant loss:", invariant_loss.item())
            if estimate_alpha == True:
                print("estimated ratio:", ratio_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = { f'{k}_unscaled': v for k, v in loss_dict_reduced.items() }
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in ['loss_CrossEntropy']}
        optimizer.zero_grad()
        losses_final = losses_ori+invariant_loss+gd_loss+losses_integ
        if estimate_alpha == True:
            losses_final = losses_final+pseudo_loss
        # if estimate_alpha == True:
        #     losses_final = losses_final+pseudo_loss+neg_loss

        losses_final.backward()

        optimizer.step()
        metric_logger.update(loss=loss_dict_reduced_scaled['loss_CrossEntropy'], **loss_dict_reduced_scaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        itertime = time.time() - start
        # profiling step timing and memory (print every step)
        '''if torch.cuda.is_available():
            torch.cuda.synchronize(device)
            current_mem_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
            current_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        else:
            current_mem_mb = 0.0
            current_peak_mb = 0.0

        step_time = time.time() - step_start_time
        total_train_time += step_time
        train_steps += 1
        peak_mem_mb = max(peak_mem_mb, current_peak_mb)

        print(
            f"[Step {step+1:04d}/{total_steps}] "
            f"time: {step_time:.4f}s | "
            f"gpu mem: {current_mem_mb:.1f} MB | "
            f"gpu peak: {current_peak_mb:.1f} MB"
        )'''

        metric_logger.log_every(step, total_steps, datatime, itertime, print_freq, header)
    # gather the stats from all processes
    avg_step_time = total_train_time / max(train_steps, 1)
    print(
        f"[Training profiling] Avg step time: {avg_step_time:.4f}s | "
        f"Peak GPU memory: {peak_mem_mb:.1f} MB"
    )
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, total_time / total_steps))
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    visual_train = Visualize_train()
    visual_train(torch.stack(original_list),torch.stack(puzzle_list),torch.stack(sample_list), torch.stack(output_list), torch.stack(target_ori_list),torch.stack(target_list), torch.stack(target_wo_cutout_list),torch.stack(output_mixed_list),torch.stack(output_original_list),torch.stack(output_puzzle_list),epoch, writer)
    
    return stats