import argparse
import datetime
import os
import shutil
import time
import numpy as np
import dataset
import mlconfig
import torch
import util
import madrys
import models
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import least_squares

class Evaluator():
    def __init__(self, data_loader, logger, config):
        self.loss_meters = util.AverageMeter()
        if args.report_seperated_avg:
            self.no_protect_acc_meters = util.AverageMeter()
            self.protect_acc_meters = util.AverageMeter()
        else:
            self.acc_meters = util.AverageMeter()
        self.criterion = torch.nn.BCEWithLogitsLoss(weight = loss_weight)
        self.data_loader = data_loader
        self.logger = logger
        self.log_frequency = config.log_frequency if config.log_frequency is not None else 100
        self.config = config
        self.current_acc = 0
        self.current_acc_top5 = 0
        # self.confusion_matrix = torch.zeros(config.num_classes, config.num_classes)
        return

    def _reset_stats(self):
        self.loss_meters = util.AverageMeter()
        if args.report_seperated_avg:
            self.no_protect_acc_meters = util.AverageMeter()
            self.protect_acc_meters = util.AverageMeter()
        else:
            self.acc_meters = util.AverageMeter()
        return

    def eval(self, epoch, model, show_all_task = False):
        model.eval()
        if args.use_auc:
            Labels = torch.tensor([])
            Preds = torch.tensor([])
            for i, (images, labels) in enumerate(self.data_loader["test_dataset"]):
                start = time.time()
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.no_grad():
                    logits = model(images)
                    loss = self.criterion(logits, labels)
                    self.loss_meters.update(loss.item(), labels.shape[0])
                    Labels = torch.cat([Labels, labels[:,:args.n_tasks].cpu()], dim=0)
                    Preds = torch.cat([Preds, torch.sigmoid(logits[:,:args.n_tasks]).detach().cpu()], dim=0)
                end = time.time()
                time_used = end - start
            auc_per_task = roc_auc_score(Labels.numpy(), Preds.numpy(), average=None)
            auc_avg = auc_per_task.mean()
            log_payload = {"loss": loss,
                       "loss_avg": self.loss_meters.avg,
                       # "auc": auc,
                       "auc_avg": auc_avg,
                       }
            self.evaluated_auc_avg = auc_avg
        else:
            if show_all_task:
                self.acc_each_meters = dict()
                for ii in range(args.n_tasks): 
                    self.acc_each_meters[ii] = util.AverageMeter()
            for i, (images, labels) in enumerate(self.data_loader["test_dataset"]):
                start = time.time()
                log_payload = self.eval_batch(images=images, labels = labels, model=model, show_all_task = show_all_task)
                end = time.time()
                time_used = end - start
        display = util.log_display(epoch=epoch,
                                   global_step=i,
                                   time_elapse=time_used,
                                   **log_payload)
        if self.logger is not None:
            self.logger.info(display)
        return

    def eval_batch(self, images, labels, model, show_all_task = False):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.no_grad():
            logits = model(images)
            # losses = self.criterion(logits, labels).mean(dim=0)
            # loss = torch.sum(losses)
            loss = self.criterion(logits, labels)

        self.loss_meters.update(loss.item(), labels.shape[0])

        if args.report_seperated_avg:
            no_protect_acc = ((logits > 0).eq(labels).float())[:,:args.no_protect_first].mean().cpu().item()
            protect_acc = ((logits > 0).eq(labels).float())[:,args.no_protect_first:args.n_tasks].mean().cpu().item()
            self.no_protect_acc_meters.update(no_protect_acc, labels.shape[0])  
            self.protect_acc_meters.update(protect_acc, labels.shape[0])  
            payload = {"loss": loss,
                   "loss_avg": self.loss_meters.avg,
                   "no_protect_acc": no_protect_acc,
                   "no_protect_acc_avg": self.no_protect_acc_meters.avg,
                   "protect_acc": protect_acc,
                   "protect_acc_avg": self.protect_acc_meters.avg,
                   }
        else:
            if show_all_task:
                acc = ((logits > 0).eq(labels).float())[:,:args.n_tasks].mean().cpu().item()
                self.acc_meters.update(acc, labels.shape[0])  
                payload = {"loss": loss,
                       "loss_avg": self.loss_meters.avg,
                       "acc": acc,
                       "acc_avg": self.acc_meters.avg,
                       }
                acc_each = dict()
                for ii in range(args.n_tasks):
                    acc_each[ii] = ((logits > 0).eq(labels).float())[:,ii].mean().cpu().item()
                    self.acc_each_meters[ii].update(acc_each[ii], labels.shape[0])
                    payload["acc_{}_avg".format(ii)] = self.acc_each_meters[ii].avg
            else:
                acc = ((logits > 0).eq(labels).float())[:,:args.n_tasks].mean().cpu().item()
                self.acc_meters.update(acc, labels.shape[0])  
                payload = {"loss": loss,
                       "loss_avg": self.loss_meters.avg,
                       "acc": acc,
                       "acc_avg": self.acc_meters.avg,
                       }
        return payload

    def eval_feats(self, epoch, model):
        self.feats = dict()
        for ii in range(args.n_tasks):
            self.feats[ii] = dict()
            self.feats[ii][0] = torch.tensor([]).to(device)
            self.feats[ii][1] = torch.tensor([]).to(device)
        model.eval()
        for i, (images, labels) in enumerate(self.data_loader["train_dataset"]):
            print (i)
            start = time.time()
            self.eval_feats_batch(images=images, labels = labels, model=model)
            end = time.time()
            time_used = end - start
        Ratio_std = []
        Ratio_variance = []
        max_Ratio_std = []
        min_Ratio_std = []
        max_Ratio_variance = []
        min_Ratio_variance = []
        Relative_variance = []
        max_Relative_variance = []
        min_Relative_variance = []
        Relative_std = []
        max_Relative_std = []
        min_Relative_std = []
        for ii in range(args.n_tasks):
            variance_0 = self.feats[ii][0].var(dim=0).squeeze()
            variance_1 = self.feats[ii][1].var(dim=0).squeeze()
            distance = (self.feats[ii][0].mean(0).abs() - self.feats[ii][1].mean(0).abs()).abs().squeeze()
            max_Ratio_std.append(((variance_0 ** 0.5 + variance_1 ** 0.5)/distance).max().cpu())
            min_Ratio_std.append(((variance_0 ** 0.5 + variance_1 ** 0.5)/distance).min().cpu())
            max_Ratio_variance.append(((variance_0 + variance_1 )/(distance**2)).max().cpu())
            min_Ratio_variance.append(((variance_0 + variance_1 )/(distance**2)).min().cpu())
            ratio_std = ((variance_0 ** 0.5 + variance_1 ** 0.5)/distance).mean()
            ratio_variance = ((variance_0 + variance_1)/(distance**2)).mean()
            Ratio_std.append(ratio_std.cpu())
            Ratio_variance.append(ratio_variance.cpu())

            mean_0 = self.feats[ii][0].mean(dim=0, keepdim=True)
            mean_1 = self.feats[ii][1].mean(dim=0, keepdim=True)
            variance_0 = self.feats[ii][0].var(dim=0, keepdim=True)
            variance_1 = self.feats[ii][1].var(dim=0, keepdim=True)
            relative_variance_0 = variance_0 / (mean_0 ** 2 + 1e-6)
            relative_variance_1 = variance_1 / (mean_1 ** 2 + 1e-6)
            Relative_variance.append((relative_variance_0 + relative_variance_1).squeeze().mean().cpu())
            max_Relative_variance.append((relative_variance_0 + relative_variance_1).squeeze().max().cpu())
            min_Relative_variance.append((relative_variance_0 + relative_variance_1).squeeze().min().cpu())
            Relative_std.append((relative_variance_0 ** 0.5 + relative_variance_1 ** 0.5).squeeze().mean().cpu())
            max_Relative_std.append((relative_variance_0 ** 0.5 + relative_variance_1 ** 0.5).squeeze().max().cpu())
            min_Relative_std.append((relative_variance_0 ** 0.5 + relative_variance_1 ** 0.5).squeeze().min().cpu())
        log_payload={
               "Ratio_std": sum(Ratio_std)/args.n_tasks,
               "max_Ratio_std": sum(max_Ratio_std)/args.n_tasks,
               "min_Ratio_std": sum(min_Ratio_std)/args.n_tasks,
               "Ratio_variance": sum(Ratio_variance)/args.n_tasks,
               "max_Ratio_variance": sum(max_Ratio_variance)/args.n_tasks,
               "min_Ratio_variance": sum(min_Ratio_variance)/args.n_tasks,
               "Relative_variance": sum(Relative_variance)/args.n_tasks,
               "max_Relative_variance": sum(max_Relative_variance)/args.n_tasks,
               "min_Relative_variance": sum(min_Relative_variance)/args.n_tasks,
               "Relative_std": sum(Relative_std)/args.n_tasks,
               "max_Relative_std": sum(max_Relative_std)/args.n_tasks,
               "min_Relative_std": sum(min_Relative_std)/args.n_tasks,
               }
        display = util.log_display(epoch=epoch,
                                   global_step=i,
                                   time_elapse=time_used,
                                   **log_payload)
        if self.logger is not None:
            self.logger.info(display)
        return

    def eval_feats_batch(self, images, labels, model):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.no_grad():
            logits, features = model(images, return_representation=True)
            loss = self.criterion(logits, labels)
            for j in range(len(images)):
                for ii in range(args.n_tasks):
                    if labels[j, ii] == 0:
                        self.feats[ii][0] = torch.cat([self.feats[ii][0], features[j].unsqueeze(0)], dim=0)
                    if labels[j, ii] == 1:
                        self.feats[ii][1] = torch.cat([self.feats[ii][1], features[j].unsqueeze(0)], dim=0)
        return


class Trainer():
    def __init__(self, criterion, data_loader, logger, config, global_step=0,
                 target='train_dataset'):
        self.criterion = criterion
        self.data_loader = data_loader
        self.logger = logger
        self.config = config
        self.log_frequency = config.log_frequency if config.log_frequency is not None else 100
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.global_step = global_step
        self.target = target
        print(self.target)

    def _reset_stats(self):
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()

    def train(self, epoch, model, criterion, optimizer):
        model.train()
        for i, (images, labels) in enumerate(self.data_loader[self.target]):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            start = time.time()
            log_payload = self.train_batch(images, labels, model, optimizer)
            end = time.time()
            time_used = end - start
            if self.global_step % self.log_frequency == 0:
                display = util.log_display(epoch=epoch,
                                           global_step=self.global_step,
                                           time_elapse=time_used,
                                           **log_payload)
                self.logger.info(display)
            self.global_step += 1
        return self.global_step

    def train_batch(self, images, labels, model, optimizer):
        model.zero_grad()
        if args.weighting == "RLW":
            batch_weight = args.n_tasks * F.softmax(torch.randn(args.n_tasks), dim=-1).to(device)
            loss_weight = torch.zeros(model.n_tasks).float().to(device)
            loss_weight[:args.n_tasks] = batch_weight
            self.criterion.weight = loss_weight

        if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss):
            optimizer.zero_grad()
            logits, features = model(images, return_representation=True)
            # if args.regularization:
            #     lambda_reg = 100
            #     loss = self.criterion(logits, labels) + lambda_reg * model.regularization_loss()
            # else:
            #     loss = self.criterion(logits, labels)
            if args.weighting == "UW":
                self.criterion.reduction = "none"
                losses = self.criterion(logits, labels).mean(dim=0)[:args.n_tasks]
                loss = (losses/(2*model.loss_scale.exp())+model.loss_scale/2).sum()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
            elif args.weighting == "Aligned_MTL":
                self.criterion.reduction = "none"
                losses = self.criterion(logits, labels).mean(dim=0)[:args.n_tasks]
                loss = torch.mul(losses, torch.ones_like(losses).to(device)).mean()
                grads = _compute_grad(losses, model, grad_index, grad_dim)
                M = torch.matmul(grads, grads.t()) # [num_tasks, num_tasks]
                # lmbda, V = torch.symeig(M, eigenvectors=True)
                lmbda, V = torch.linalg.eigh(M)
                tol = (torch.max(lmbda)* max(M.shape[-2:])* torch.finfo().eps)
                rank = sum(lmbda > tol)
                order = torch.argsort(lmbda, dim=-1, descending=True)
                lmbda, V = lmbda[order][:rank], V[:, order][:, :rank]
                sigma = torch.diag(1 / lmbda.sqrt())
                B = lmbda[-1].sqrt() * ((V @ sigma) @ V.t())
                alpha = B.sum(0)
                new_grads = sum([alpha[i] * grads[i] for i in range(args.n_tasks)])
                count = 0
                for param in model.shared_parameters():
                    if param.grad is not None:
                        beg = 0 if count == 0 else sum(grad_index[:count])
                        end = sum(grad_index[:(count+1)])
                        param.grad.data = new_grads[beg:end].contiguous().view(param.data.size()).data.clone()
                    count += 1
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
            elif args.weighting == "FairGrad":
                self.criterion.reduction = "none"
                losses = self.criterion(logits, labels).mean(dim=0)[:args.n_tasks]
                alpha = 1.0
                grads = _compute_grad(losses, model, grad_index, grad_dim, mode='autograd')
                GTG = torch.mm(grads, grads.t())
                x_start = np.ones(args.n_tasks) / args.n_tasks
                A = GTG.data.cpu().numpy()
                def objfn(x):
                    return np.dot(A, x) - np.power(1 / x, 1 / alpha)
                res = least_squares(objfn, x_start, bounds=(0, np.inf))
                w_cpu = res.x
                ww = torch.Tensor(w_cpu).to(device)
                loss = torch.sum(ww*losses)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
            else:
                loss = self.criterion(logits, labels)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
        else:
            logits, loss = self.criterion(model, images, labels, optimizer)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
        optimizer.step()
        
        acc = ((logits > 0).eq(labels).float())[:,:args.n_tasks].mean().cpu().item()
        
        self.acc_meters.update(acc, labels.shape[0])
        self.loss_meters.update(loss.item(), labels.shape[0])

        payload = {"loss": loss,
                   "loss_avg": self.loss_meters.avg,
                   "acc": acc,
                   "acc_avg": self.acc_meters.avg,
                   "lr": optimizer.param_groups[0]['lr'],
                   "|gn|": grad_norm}
        return payload

mlconfig.register(madrys.MadrysLoss)
mlconfig.register(madrys.MultiMadrysLoss)
mlconfig.register(madrys.BCEMadrysLoss)

# General Options
parser = argparse.ArgumentParser(description='ClasswiseNoise')
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--version', type=str, default="resnet18")
parser.add_argument('--exp_name', type=str, default="test_exp")
parser.add_argument('--config_path', type=str, default='configs/cifar10')
parser.add_argument('--load_model', action='store_true', default=False)
parser.add_argument('--data_parallel', action='store_true', default=False)
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--save_frequency', default=-1, type=int)
# Datasets Options
parser.add_argument('--train_face', action='store_true', default=False)
parser.add_argument('--train_portion', default=1.0, type=float)
parser.add_argument('--train_batch_size', default=128, type=int, help='perturb step size')
parser.add_argument('--eval_batch_size', default=256, type=int, help='perturb step size')
parser.add_argument('--num_of_workers', default=4, type=int, help='workers for loader')
parser.add_argument('--train_data_type', type=str, default='CIFAR10')
parser.add_argument('--test_data_type', type=str, default='CIFAR10')
parser.add_argument('--train_data_path', type=str, default='../datasets')
parser.add_argument('--test_data_path', type=str, default='../datasets')
parser.add_argument('--perturb_type', default='classwise', type=str, help='Perturb type')
parser.add_argument('--patch_location', default='center', type=str, choices=['center', 'random'], help='Location of the noise')
parser.add_argument('--poison_rate', default=1.0, type=float)
parser.add_argument('--perturb_tensor_filepath', default=None, type=str)
parser.add_argument('--identity', default='race', type=str, choices=['race', 'age', 'gender', 'class', 'bias'])

parser.add_argument('--jpeg', action='store_true', default=False)
parser.add_argument('--bdr', action='store_true', default=False)
parser.add_argument('--grayscale', action='store_true', default=False)
parser.add_argument('--poison_idxs_filepath', default=None, type=str)

parser.add_argument('--n_tasks', default=40, type=int)
parser.add_argument('--regularization', action='store_true', default=False)
parser.add_argument('--use_auc', action='store_true', default=False)

parser.add_argument('--weighting', default='LS', type=str)
parser.add_argument('--report_seperated_avg', action='store_true', default=False)
parser.add_argument('--no_protect_first', default=10, type=int)

parser.add_argument('--partial_rate', default=1.0, type=float)
global args
args = parser.parse_args()


# Set up Experiments
if args.exp_name == '':
    args.exp_name = 'exp_' + datetime.datetime.now()

exp_path = os.path.join(args.exp_name, args.version)
log_file_path = os.path.join(exp_path, args.version)
checkpoint_path = os.path.join(exp_path, 'checkpoints')
checkpoint_path_file = os.path.join(checkpoint_path, args.version)
util.build_dirs(exp_path)
util.build_dirs(checkpoint_path)
logger = util.setup_logger(name=args.version, log_file=log_file_path + ".log")

# CUDA Options
logger.info("PyTorch Version: %s" % (torch.__version__))
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
    logger.info("GPU List: %s" % (device_list))
else:
    device = torch.device('cpu')

# Load Exp Configs
config_file = os.path.join(args.config_path, args.version)+'.yaml'
config = mlconfig.load(config_file)
config.set_immutable()
for key in config:
    logger.info("%s: %s" % (key, config[key]))
shutil.copyfile(config_file, os.path.join(exp_path, args.version+'.yaml'))


def train(starting_epoch, model, optimizer, scheduler, criterion, trainer, evaluator, ENV, data_loader):
    for epoch in range(starting_epoch, config.epochs):
        logger.info("")
        logger.info("="*20 + "Training Epoch %d" % (epoch) + "="*20)

        # Train
        ENV['global_step'] = trainer.train(epoch, model, criterion, optimizer)
        # ENV['train_history'].append(trainer.acc_meters.avg*100)
        scheduler.step()

        # Eval
        logger.info("="*20 + "Eval Epoch %d" % (epoch) + "="*20)
        is_best = False
        if not args.train_face:
            evaluator.eval(epoch, model)
            if args.use_auc:
                payload = ('Eval Loss:%.4f\tEval auc_avg: %.2f' % (evaluator.loss_meters.avg, evaluator.evaluated_auc_avg*100))
            else:
                if args.report_seperated_avg:
                    payload = ('Eval Loss:%.4f\tEval no_protect_acc: %.2f\tEval protect_acc: %.2f' % (evaluator.loss_meters.avg, evaluator.no_protect_acc_meters.avg*100, evaluator.protect_acc_meters.avg*100))
                else:
                    payload = ('Eval Loss:%.4f\tEval acc: %.2f' % (evaluator.loss_meters.avg, evaluator.acc_meters.avg*100))
            logger.info(payload)
            # ENV['eval_history'].append(evaluator.acc_meters.avg*100)
            # ENV['curren_acc'] = evaluator.acc_meters.avg*100
            # ENV['cm_history'].append(evaluator.confusion_matrix.cpu().numpy().tolist())
            # Reset Stats
            trainer._reset_stats()
            evaluator._reset_stats()
        else:
            pass
            

        # Save Model
        target_model = model.module if args.data_parallel else model
        util.save_model(ENV=ENV,
                        epoch=epoch,
                        model=target_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        is_best=is_best,
                        filename=checkpoint_path_file)
        logger.info('Model Saved at %s', checkpoint_path_file)

        if args.save_frequency > 0 and epoch % args.save_frequency == 0:
            filename = checkpoint_path_file + '_epoch%d' % (epoch)
            util.save_model(ENV=ENV,
                            epoch=epoch,
                            model=target_model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            filename=filename)
            logger.info('Model Saved at %s', filename)

    return

def _compute_grad(losses, model, grad_index, grad_dim, mode = 'backward'):
    grads = torch.zeros(args.n_tasks, grad_dim).to(device)
    for tn in range(args.n_tasks):
        if mode == 'backward':
            losses[tn].backward(retain_graph=True) if (tn+1)!=args.n_tasks else losses[tn].backward()
            grad = torch.zeros(grad_dim).to(device)
            count = 0
            for param in model.shared_parameters():
                if param.grad is not None:
                    beg = 0 if count == 0 else sum(grad_index[:count])
                    end = sum(grad_index[:(count+1)])
                    grad[beg:end] = param.grad.data.view(-1)
                count += 1
            grads[tn] = grad
        elif mode == 'autograd':
            grad = list(torch.autograd.grad(losses[tn], model.shared_parameters(), retain_graph=True))
            grads[tn] = torch.cat([g.view(-1) for g in grad])
        for param in model.shared_parameters():
            if param.grad is not None: 
                param.grad.zero_()
    return grads

def main():
    model = config.model().to(device)
    # print (model)
    datasets_generator = config.dataset(train_data_type=args.train_data_type,
                                        train_data_path=args.train_data_path,
                                        test_data_type=args.test_data_type,
                                        test_data_path=args.test_data_path,
                                        train_batch_size=args.train_batch_size,
                                        eval_batch_size=args.eval_batch_size,
                                        num_of_workers=args.num_of_workers,
                                        poison_rate=args.poison_rate,
                                        perturb_type=args.perturb_type,
                                        patch_location=args.patch_location,
                                        perturb_tensor_filepath=args.perturb_tensor_filepath,
                                        seed=args.seed, identity = args.identity, jpeg = args.jpeg, bdr = args.bdr, grayscale = args.grayscale, partial_rate = args.partial_rate)
    logger.info('Training Dataset: %s' % str(datasets_generator.datasets['train_dataset']))
    logger.info('Test Dataset: %s' % str(datasets_generator.datasets['test_dataset']))
    if 'Poison' in args.train_data_type:
        with open(os.path.join(exp_path, 'poison_targets.npy'), 'wb') as f:
            if not (isinstance(datasets_generator.datasets['train_dataset'], dataset.MixUp) or isinstance(datasets_generator.datasets['train_dataset'], dataset.CutMix)):
                poison_targets = np.array(datasets_generator.datasets['train_dataset'].poison_samples_idx)
                np.save(f, poison_targets)
                logger.info(poison_targets)
                logger.info('Poisoned: %d/%d' % (len(poison_targets), len(datasets_generator.datasets['train_dataset'])))
                logger.info('Poisoned samples idx saved at %s' % (os.path.join(exp_path, 'poison_targets')))
                logger.info('Poisoned Class %s' % (str(datasets_generator.datasets['train_dataset'].poison_class)))

    if args.train_portion == 1.0:
        data_loader = datasets_generator.getDataLoader()
        train_target = 'train_dataset'
    else:
        train_target = 'train_subset'
        data_loader = datasets_generator._split_validation_set(args.train_portion,
                                                               train_shuffle=True,
                                                               train_drop_last=True)

    global loss_weight
    logger.info("param size = %fMB", util.count_parameters_in_MB(model))
    if args.weighting == "UW":
        model.loss_scale = nn.Parameter(torch.tensor([-0.5]*args.n_tasks, device=device))
    if args.weighting in ["Aligned_MTL", "FairGrad"]:
        global grad_index, grad_dim
        grad_index = []
        for param in model.shared_parameters():
            grad_index.append(param.data.numel())
        grad_dim = sum(grad_index)
    if ('vit_b' in args.exp_name) or (args.weighting == "UW"):
        optimizer = config.optimizer(model.parameters())
    else:    
        optimizer = config.optimizer([dict(params=model.shared_parameters()), dict(params=model.task_specific_parameters())])
    scheduler = config.scheduler(optimizer)
    criterion = config.criterion()
    loss_weight = torch.zeros(model.n_tasks).float().cuda()
    loss_weight[:args.n_tasks] = 1
    criterion.weight = loss_weight
    if 'pos_weight' in config:
        print ('use_balanced_weight')
        criterion.pos_weight = torch.tensor(config['pos_weight']).to(device)
    trainer = Trainer(criterion, data_loader, logger, config, target=train_target)
    evaluator = Evaluator(data_loader, logger, config)

    starting_epoch = 0
    ENV = {'global_step': 0,
           'best_acc': 0.0,
           'curren_acc': 0.0,
           'best_pgd_acc': 0.0,
           'train_history': [],
           'eval_history': [],
           'pgd_eval_history': [],
           'genotype_list': [],
           'cm_history': []}

    if args.load_model:
        checkpoint = util.load_model(filename=checkpoint_path_file,
                                     model=model,
                                     optimizer=optimizer,
                                     alpha_optimizer=None,
                                     scheduler=scheduler)
        starting_epoch = checkpoint['epoch']
        ENV = checkpoint['ENV']
        trainer.global_step = ENV['global_step']
        logger.info("File %s loaded!" % (checkpoint_path_file))

    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    
    if args.poison_idxs_filepath is not None:
        poison_idxs = np.load(args.poison_idxs_filepath).tolist()
        data_loader["train_dataset"].dataset.data = data_loader["train_dataset"].dataset.data[poison_idxs]
        data_loader["train_dataset"].dataset.targets = [data_loader["train_dataset"].dataset.targets[idx] for idx in poison_idxs]
    
    if args.train:
        train(starting_epoch, model, optimizer, scheduler, criterion, trainer, evaluator, ENV, data_loader)
    else:
        evaluator.eval(starting_epoch, model, show_all_task = True)
        payload = {"loss_avg": evaluator.loss_meters.avg,
                   "acc_avg": evaluator.acc_meters.avg}
        for ii in range(args.n_tasks):
            payload["acc_{}_avg".format(ii)] = evaluator.acc_each_meters[ii].avg
        logger.info(payload)


if __name__ == '__main__':
    for arg in vars(args):
        logger.info("%s: %s" % (arg, getattr(args, arg)))
    start = time.time()
    main()
    end = time.time()
    cost = (end - start) / 86400
    payload = "Running Cost %.2f Days \n" % cost
    logger.info(payload)
