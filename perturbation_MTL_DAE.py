import argparse
import collections
import datetime
import os
import shutil
import time
import dataset
import mlconfig
import toolbox
import torch
import util
import madrys
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable

class Evaluator():
    def __init__(self, criterion, data_loader, logger, config):
        self.loss_meters = util.AverageMeter()
        self.acc_all_meters = util.AverageMeter()
        self.acc_meters = dict()
        for i in range(args.n_tasks):
            self.acc_meters[i] = util.AverageMeter()
        self.criterion = criterion
        self.data_loader = data_loader
        self.logger = logger
        self.log_frequency = config.log_frequency if config.log_frequency is not None else 4
        self.config = config
        self.current_acc = 0
        self.current_acc_top5 = 0
        # self.confusion_matrix = torch.zeros(config.num_classes, config.num_classes)
        return

    def _reset_stats(self):
        self.loss_meters = util.AverageMeter()
        self.acc_all_meters = util.AverageMeter()
        self.acc_meters = dict()
        for i in range(args.n_tasks):
            self.acc_meters[i] = util.AverageMeter()
        return

    def eval(self, epoch, model):
        model.eval()
        for i, (images, labels) in enumerate(self.data_loader["test_dataset"]):
            start = time.time()
            log_payload = self.eval_batch(images=images, labels = labels, model=model)
            end = time.time()
            time_used = end - start
        display = util.log_display(epoch=epoch,
                                   global_step=i,
                                   time_elapse=time_used,
                                   **log_payload)
        if self.logger is not None:
            self.logger.info(display)
        return

    def eval_batch(self, images, labels, model):
        images = images.to(device, non_blocking=True)
        labels = [label.to(device, non_blocking=True) for label in labels]
        with torch.no_grad():
            logits = model(images)
            loss = self.criterion(logits, labels)
            
        acc_metric = MultiAccuracy()
        acc_list = acc_metric(logits[:args.n_tasks], labels[:args.n_tasks])
        acc_all = sum(acc_list)/len(acc_list)
        self.acc_all_meters.update(acc_all, images.shape[0])
        self.loss_meters.update(loss.item(), images.shape[0])
        for i in range(args.n_tasks):
            self.acc_meters[i].update(acc_list[i], images.shape[0])

        payload = dict()
        payload["loss"] = loss
        payload["loss_avg"] = self.loss_meters.avg
        payload["acc_all"] = acc_all
        payload["acc_all_avg"] = self.acc_all_meters.avg
        for i, acc in enumerate(acc_list):
            payload["acc_{}".format(i)] = acc
            payload["acc_{}_avg".format(i)] = self.acc_meters[i].avg
        return payload


class Trainer():
    def __init__(self, criterion, data_loader, logger, config, global_step=0,
                 target='train_dataset'):
        self.criterion = criterion
        self.data_loader = data_loader
        self.logger = logger
        self.config = config
        self.log_frequency = config.log_frequency if config.log_frequency is not None else 40
        self.loss_meters = util.AverageMeter()
        self.acc_all_meters = util.AverageMeter()
        self.acc_meters = dict()
        for i in range(args.n_tasks):
            self.acc_meters[i] = util.AverageMeter()
        self.global_step = global_step
        self.target = target
        print(self.target)

    def _reset_stats(self):
        self.loss_meters = util.AverageMeter()
        self.acc_all_meters = util.AverageMeter()
        self.acc_meters = dict()
        for i in range(args.n_tasks):
            self.acc_meters[i] = util.AverageMeter()

    def train(self, epoch, model, criterion, optimizer):
        model.train()
        for i, (images, labels) in enumerate(self.data_loader[self.target]):
            images = images.to(device, non_blocking=True)
            labels = [label.to(device, non_blocking=True) for label in labels]
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

    def train_batch(self, images, labels, model, optimizer, transform):
        model.zero_grad()
        optimizer.zero_grad()
        if transform is not None:
            logits = model(transform(images))
        else:
            logits = model(images)

        loss = self.criterion(logits, labels)
            
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
        optimizer.step()
        
        # acc = ((logits > 0).eq(labels).float())[:,:args.n_tasks].mean().cpu().item()
        acc_metric = MultiAccuracy()
        acc_list = acc_metric(logits[:args.n_tasks], labels[:args.n_tasks])
        acc_all = sum(acc_list)/len(acc_list)
        self.acc_all_meters.update(acc_all, images.shape[0])
        self.loss_meters.update(loss.item(), images.shape[0])
        for i in range(args.n_tasks):
            self.acc_meters[i].update(acc_list[i], images.shape[0])

        payload = dict()
        payload["loss"] = loss
        payload["loss_avg"] = self.loss_meters.avg
        payload["acc_all"] = acc_all
        payload["acc_all_avg"] = self.acc_all_meters.avg
        for i, acc in enumerate(acc_list):
            payload["acc_{}".format(i)] = acc
            payload["acc_{}_avg".format(i)] = self.acc_meters[i].avg
        payload["lr"] = optimizer.param_groups[0]['lr']
        payload["|gn|"] = grad_norm
        return payload

    def adv_train_batch(self, images, labels, model, optimizer, transform):
        model.eval()
        x_natural = images.detach()
        epsilon = args.epsilon/2
        alpha = args.step_size/2
        steps = args.num_steps
        x_adv = x_natural + 0.001 * torch.randn_like(x_natural).detach()
        x_adv = x_adv.clamp(0.0, 1.0).detach().requires_grad_()
        for _ in range(steps):
            # logits_adv = model(x_adv)
            if transform is not None:
                logits_adv = model(transform(x_adv))
            else:
                logits_adv = model(x_adv)
            loss_adv = self.criterion(logits_adv, labels)
            grad = torch.autograd.grad(loss_adv, [x_adv])[0]
            x_adv = x_adv.detach() + alpha * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = x_adv.clamp(0.0, 1.0).detach().requires_grad_()

        model.train()  # switch back to train mode
        model.zero_grad()
        optimizer.zero_grad()
        # logits = model(x_adv)
        if transform is not None:
            logits = model(transform(x_adv))
        else:
            logits = model(x_adv)
        loss = self.criterion(logits, labels)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
        optimizer.step()
        acc_metric = MultiAccuracy()
        acc_list = acc_metric(logits[:args.n_tasks], labels[:args.n_tasks])
        acc_all = sum(acc_list) / len(acc_list)
        self.acc_all_meters.update(acc_all, images.shape[0])
        self.loss_meters.update(loss.item(), images.shape[0])
        for i in range(args.n_tasks):
            self.acc_meters[i].update(acc_list[i], images.shape[0])
        payload = {
            "loss": loss,
            "loss_avg": self.loss_meters.avg,
            "acc_all": acc_all,
            "acc_all_avg": self.acc_all_meters.avg,
            "lr": optimizer.param_groups[0]['lr'],
            "|gn|": grad_norm
        }
        for i, acc in enumerate(acc_list):
            payload[f"acc_{i}"] = acc
            payload[f"acc_{i}_avg"] = self.acc_meters[i].avg
        return payload

class MultiAccuracy:
    def __init__(self, top_k=1):
        self.top_k = top_k
    def __call__(self, logits_list, labels_list):
        accuracy_list = []
        for logits, labels in zip(logits_list, labels_list):
            _, pred = logits.topk(self.top_k, dim=1, largest=True, sorted=True)
            correct = pred.eq(labels.view(-1, 1).expand_as(pred))
            accuracy = correct.sum().item() / labels.size(0)
            accuracy_list.append(accuracy)
        return accuracy_list

mlconfig.register(madrys.MadrysLoss)

# General Options
parser = argparse.ArgumentParser(description='ClasswiseNoise')
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--version', type=str, default="resnet18")
parser.add_argument('--exp_name', type=str, default="test_exp")
parser.add_argument('--config_path', type=str, default='configs/cifar10')
parser.add_argument('--load_model', action='store_true', default=False)
parser.add_argument('--data_parallel', action='store_true', default=False)
# Datasets Options
parser.add_argument('--train_batch_size', default=512, type=int, help='perturb step size')
parser.add_argument('--eval_batch_size', default=512, type=int, help='perturb step size')
parser.add_argument('--num_of_workers', default=8, type=int, help='workers for loader')
parser.add_argument('--train_data_type', type=str, default='CIFAR10')
parser.add_argument('--train_data_path', type=str, default='../datasets')
parser.add_argument('--test_data_type', type=str, default='CIFAR10')
parser.add_argument('--test_data_path', type=str, default='../datasets')
# Perturbation Options
parser.add_argument('--universal_train_portion', default=0.2, type=float)
parser.add_argument('--universal_stop_error', default=0.5, type=float)
parser.add_argument('--universal_train_target', default='train_subset', type=str)
parser.add_argument('--train_step', default=10, type=int)
parser.add_argument('--use_subset', action='store_true', default=False)
parser.add_argument('--attack_type', default='min-min', type=str, choices=['min-min', 'min-min-max', 'min-max', 'random', 'mix', 'sep', 'sep_fa'], help='Attack type')
parser.add_argument('--perturb_type', default='samplewise', type=str, choices=['classwise', 'samplewise'], help='Perturb type')
parser.add_argument('--patch_location', default='center', type=str, choices=['center', 'random'], help='Location of the noise')
parser.add_argument('--epsilon', default=8, type=float, help='perturbation')
parser.add_argument('--num_steps', default=1, type=int, help='perturb number of steps')
parser.add_argument('--step_size', default=0.8, type=float, help='perturb step size')
parser.add_argument('--random_start', action='store_true', default=False)
parser.add_argument('--identity', default='race', type=str, choices=['race', 'age', 'gender', 'class', 'bias'])

parser.add_argument('--n_tasks', default=40, type=int)
parser.add_argument('--load_model_path', type=str, default='experiments_MTL/CelebA/clean_MTL/resnet18/checkpoints/resnet18')
parser.add_argument('--regularization', action='store_true', default=False)
parser.add_argument('--embedding_regularization', action='store_true', default=False)
parser.add_argument('--cross_embedding_regularization', action='store_true', default=False)
parser.add_argument('--stop_epoch', type=int, default=60)
global args
args = parser.parse_args()

# Convert Eps
args.epsilon = args.epsilon / 255
args.step_size = args.step_size / 255

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


config_s_file = os.path.join(args.config_path, 'resnet18DAE')+'.yaml'
config_s = mlconfig.load(config_s_file)
config_s.set_immutable()


def train(starting_epoch, model, optimizer, scheduler, criterion, trainer, evaluator, ENV, data_loader):
    for epoch in range(starting_epoch, config.epochs):
        logger.info("")
        logger.info("="*20 + "Training Epoch %d" % (epoch) + "="*20)

        # Train
        ENV['global_step'] = trainer.train(epoch, model, criterion, optimizer)
        ENV['train_history'].append(trainer.acc_meters.avg*100)
        scheduler.step()

        # Eval
        logger.info("="*20 + "Eval Epoch %d" % (epoch) + "="*20)
        evaluator.eval(epoch, model)
        payload = ('Eval Loss:%.4f\tEval acc: %.2f' % (evaluator.loss_meters.avg, evaluator.acc_meters.avg*100))
        logger.info(payload)
        ENV['eval_history'].append(evaluator.acc_meters.avg*100)
        ENV['curren_acc'] = evaluator.acc_meters.avg*100

        # Reset Stats
        trainer._reset_stats()
        evaluator._reset_stats()

        # Save Model
        target_model = model.module if args.data_parallel else model
        util.save_model(ENV=ENV,
                        epoch=epoch,
                        model=target_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        filename=checkpoint_path_file)
        logger.info('Model Saved at %s', checkpoint_path_file)
    return


def universal_perturbation_eval(noise_generator, random_noise, data_loader, model, eval_target=args.universal_train_target):
    loss_meter = util.AverageMeter()
    err_meter = util.AverageMeter()
    random_noise = random_noise.to(device)
    model = model.to(device)
    for i, (images, labels) in enumerate(data_loader[eval_target]):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if random_noise is not None:
            for i in range(len(labels)):
                class_index = labels[i].item()
                noise = random_noise[class_index]
                mask_cord, class_noise = noise_generator._patch_noise_extend_to_img(noise, image_size=images[i].shape, patch_location=args.patch_location)
                images[i] += class_noise
        pred = model(images)
        err = (pred.data.max(1)[1] != labels.data).float().sum()
        loss = torch.nn.CrossEntropyLoss()(pred, labels)
        loss_meter.update(loss.item(), len(labels))
        err_meter.update(err / len(labels))
    return loss_meter.avg, err_meter.avg


def universal_perturbation(noise_generator, trainer, evaluator, model, criterion, optimizer, scheduler, random_noise, ENV):
    # Class-Wise perturbation
    # Generate Data loader
    datasets_generator = dataset.DatasetGenerator(train_batch_size=args.train_batch_size,
                                                  eval_batch_size=args.eval_batch_size,
                                                  train_data_type=args.train_data_type,
                                                  train_data_path=args.train_data_path,
                                                  test_data_type=args.test_data_type,
                                                  test_data_path=args.test_data_path,
                                                  num_of_workers=args.num_of_workers,
                                                  seed=args.seed, no_train_augments=True, identity = args.identity)

    if args.use_subset:
        data_loader = datasets_generator._split_validation_set(train_portion=args.universal_train_portion,
                                                               train_shuffle=True, train_drop_last=True)
    else:
        data_loader = datasets_generator.getDataLoader(train_shuffle=True, train_drop_last=True)

    condition = True
    data_iter = iter(data_loader['train_dataset'])
    logger.info('=' * 20 + 'Searching Universal Perturbation' + '=' * 20)
    if hasattr(model, 'classify'):
        model.classify = True
    while condition:
        if args.attack_type == 'min-min' and not args.load_model:
            # Train Batch for min-min noise
            for j in range(0, args.train_step):
                try:
                    (images, labels) = next(data_iter)
                except:
                    data_iter = iter(data_loader['train_dataset'])
                    (images, labels) = next(data_iter)

                images, labels = images.to(device), labels.to(device)
                # Add Class-wise Noise to each sample
                train_imgs = []
                for i, (image, label) in enumerate(zip(images, labels)):
                    noise = random_noise[label.item()]
                    mask_cord, class_noise = noise_generator._patch_noise_extend_to_img(noise, image_size=image.shape, patch_location=args.patch_location)
                    train_imgs.append(images[i]+class_noise)
                # Train
                model.train()
                for param in model.parameters():
                    param.requires_grad = True
                trainer.train_batch(torch.stack(train_imgs).to(device), labels, model, optimizer)

        for i, (images, labels) in tqdm(enumerate(data_loader[args.universal_train_target]), total=len(data_loader[args.universal_train_target])):
            images, labels, model = images.to(device), labels.to(device), model.to(device)
            # Add Class-wise Noise to each sample
            batch_noise, mask_cord_list = [], []
            for i, (image, label) in enumerate(zip(images, labels)):
                noise = random_noise[label.item()]
                mask_cord, class_noise = noise_generator._patch_noise_extend_to_img(noise, image_size=image.shape, patch_location=args.patch_location)
                batch_noise.append(class_noise)
                mask_cord_list.append(mask_cord)

            # Update universal perturbation
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

            batch_noise = torch.stack(batch_noise).to(device)
            if args.attack_type == 'min-min':
                perturb_img, eta = noise_generator.min_min_attack(images, labels, model, optimizer, criterion, random_noise=batch_noise)
            elif args.attack_type == 'min-max':
                perturb_img, eta = noise_generator.min_max_attack(images, labels, model, optimizer, criterion, random_noise=batch_noise)
            else:
                raise('Invalid attack')

            class_noise_eta = collections.defaultdict(list)
            for i in range(len(eta)):
                x1, x2, y1, y2 = mask_cord_list[i]
                delta = eta[i][:, x1: x2, y1: y2]
                class_noise_eta[labels[i].item()].append(delta.detach().cpu())

            for key in class_noise_eta:
                delta = torch.stack(class_noise_eta[key]).mean(dim=0) - random_noise[key]
                class_noise = random_noise[key]
                class_noise += delta
                random_noise[key] = torch.clamp(class_noise, -args.epsilon, args.epsilon)

        # Eval termination conditions
        loss_avg, error_rate = universal_perturbation_eval(noise_generator, random_noise, data_loader, model, eval_target=args.universal_train_target)
        logger.info('Loss: {:.4f} Acc: {:.2f}%'.format(loss_avg, 100 - error_rate*100))
        random_noise = random_noise.detach()
        ENV['random_noise'] = random_noise
        if args.attack_type == 'min-min':
            condition = error_rate > args.universal_stop_error
        elif args.attack_type == 'min-max':
            condition = error_rate < args.universal_stop_error
    return random_noise


def samplewise_perturbation_eval(criterion, model_s, data_loader, model, eval_target='train_dataset', transform = None):
    loss_meter = util.AverageMeter()
    err_meter = util.AverageMeter()
    # random_noise = random_noise.to(device)
    model = model.to(device)
    idx = 0
    for i, (images, labels) in enumerate(data_loader[eval_target]):
        images = images.to(device, non_blocking=True)
        labels = [label.to(device, non_blocking=True) for label in labels]
        with torch.no_grad():
            noise = model_s(images, labels, binary = False)
            images = images + noise
        if transform is not None:
            logits = model(transform(images))
        else:
            logits = model(images)
        if args.attack_type in ['min-max', 'sep']:
            labels = [(label + model.task_outputs[i]//2) % model.task_outputs[i] for i, label in enumerate(labels)]
        loss = criterion(logits, labels)
        acc_metric = MultiAccuracy()
        acc_list = acc_metric(logits[:args.n_tasks], labels[:args.n_tasks])
        acc_all = sum(acc_list)/len(acc_list)
        err = 1 - acc_all
        loss_meter.update(loss.item(), images.shape[0])
        err_meter.update(err, images.shape[0])
    return loss_meter.avg, err_meter.avg


def sample_wise_perturbation(trainer, evaluator, model, criterion, optimizer, scheduler, model_s, optimizer_s, ENV):
    datasets_generator = dataset.DatasetGenerator(train_batch_size=args.train_batch_size,
                                                  eval_batch_size=args.eval_batch_size,
                                                  train_data_type=args.train_data_type,
                                                  train_data_path=args.train_data_path,
                                                  test_data_type=args.test_data_type,
                                                  test_data_path=args.test_data_path,
                                                  num_of_workers=args.num_of_workers,
                                                  seed=args.seed, no_train_augments=False, identity = args.identity)
    transform = datasets_generator.datasets['train_dataset'].transform
    transform.transforms.pop()
    print (transform)
    datasets_generator.datasets['train_dataset'].w_transform = False
    if args.train_data_type == 'ImageNetMini' and args.perturb_type == 'samplewise':
        data_loader = datasets_generator._split_validation_set(0.2, train_shuffle=False, train_drop_last=False)
        data_loader['train_dataset'] = data_loader['train_subset']
    else:
        data_loader = datasets_generator.getDataLoader(train_shuffle=True, train_drop_last=False)
    
    from torch.utils.data import DataLoader
    datasets_generator_noise = dataset.DatasetGenerator(train_batch_size=args.train_batch_size,
                                                  eval_batch_size=args.eval_batch_size,
                                                  train_data_type=args.train_data_type,
                                                  train_data_path=args.train_data_path,
                                                  test_data_type=args.test_data_type,
                                                  test_data_path=args.test_data_path,
                                                  num_of_workers=args.num_of_workers,
                                                  seed=args.seed, no_train_augments=False, identity = args.identity)
    datasets_generator_noise.datasets['train_dataset'].w_transform = False
    data_loader_noise = DataLoader(dataset=datasets_generator_noise.datasets['train_dataset'],
                                   batch_size=datasets_generator.train_batch_size,
                                   shuffle=False, pin_memory=True,
                                   drop_last=False, num_workers=datasets_generator.num_of_workers)

    condition = True
    data_iter = iter(data_loader['train_dataset'])
    logger.info('=' * 20 + 'Optimizating Samplewise Perturbator' + '=' * 20)
    generation_steps = 0
    while condition:
        # Search For Noise
        model_s.train()
        idx = 0
        for i, (images, labels) in enumerate(data_loader['train_dataset']):
            model = model.to(device)
            images = images.to(device, non_blocking=True)
            labels = [label.to(device, non_blocking=True) for label in labels]

            # Update sample-wise perturbation
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            
            model_s.zero_grad()
            optimizer_s.zero_grad()
            noise = model_s(images, labels, binary = False)
            perturb_img = images + noise
            if args.attack_type == 'min-min-max':
                perturb_img = perturb_img.detach()
                if transform is not None:
                    augmented_img = transform(images)
                random_noise_max = torch.FloatTensor(*augmented_img.shape).uniform_(-args.epsilon/2, args.epsilon/2).to(device).requires_grad_()
                for _ in range(args.num_steps):
                    model.zero_grad()
                    if transform is not None:
                        augmented_perturb_img = transform(perturb_img)
                        logits = model(augmented_perturb_img+random_noise_max)
                    else:
                        logits = model(perturb_img+random_noise_max)
                    loss = criterion(logits, labels)
                    grad = torch.autograd.grad(loss, random_noise_max, retain_graph=True)[0]
                    eta = args.step_size/2 * grad.data.sign()
                    random_noise_max = Variable(random_noise_max.data + eta, requires_grad=True)
                    random_noise_max = Variable(torch.clamp(random_noise_max, -args.epsilon/2, args.epsilon/2), requires_grad=True)
                random_noise_max = random_noise_max.clone().detach()
                model.zero_grad()
                model_s.zero_grad()
                optimizer_s.zero_grad()
                noise = model_s(images, labels, binary = False)
                perturb_img = images + noise
                if transform is not None:
                    augmented_perturb_img = transform(perturb_img)
                    logits = model(augmented_perturb_img+random_noise_max)
                else:
                    logits = model(perturb_img+random_noise_max)
                loss = criterion(logits, labels)
            else:
                if transform is not None:
                    augmented_perturb_img = transform(perturb_img)
                    logits = model(augmented_perturb_img)
                else:
                    logits = model(perturb_img)
                if args.attack_type == 'min-min':
                    loss = criterion(logits, labels)
                elif args.attack_type == 'min-max':
                    y = [(label + model.task_outputs[i]//2) % model.task_outputs[i] for i, label in enumerate(labels)]
                    loss = criterion(logits, y)
            loss_optimization = loss
            if args.embedding_regularization:
                loss_ER_weight = 10
                loss_ER = model_s.embedding_cosine_similarity_loss()
                loss_optimization = loss_optimization + loss_ER_weight * loss_ER
            if args.cross_embedding_regularization:
                loss_CER_weight = 200
                loss_CER = model_s.cross_embedding_cosine_similarity_loss()
                loss_optimization = loss_optimization + loss_CER_weight * loss_CER
            loss_optimization.backward()
            optimizer_s.step()

            if i % 10 == 0:
                if 'min-max' in args.attack_type:
                    labels = [(label + model.task_outputs[i]//2) % model.task_outputs[i] for i, label in enumerate(labels)]
                acc_metric = MultiAccuracy()
                acc_list = acc_metric(logits[:args.n_tasks], labels[:args.n_tasks])
                acc_all = sum(acc_list)/len(acc_list)
                output_info = 'Loss: {:.4f} '.format(loss)
                if args.embedding_regularization:
                    output_info = output_info + 'loss_ER: {:.4f} '.format(loss_ER)
                if args.cross_embedding_regularization:
                    output_info = output_info + 'Loss_CER: {:.4f} '.format(loss_CER)
                output_info = output_info + 'Acc_all: {:.2f}%'.format(acc_all*100)
                for i, acc in enumerate(acc_list):
                    output_info = output_info + ' Acc_{}: {:.2f}%'.format(i, acc*100)
                logger.info(output_info)
                # if args.embedding_regularization:
                #     logger.info('Loss: {:.4f} Loss_regularization: {:.4f} Acc: {:.2f}%'.format(loss, loss_ER, acc*100))
                # else:
                #     logger.info('Loss: {:.4f} Acc: {:.2f}%'.format(loss, acc*100))
        
        scheduler_s.step()

        model_s.eval()
        if 'min-max' in args.attack_type and not args.load_model:
            # Train Batch for min-min noise
            for j in tqdm(range(0, args.train_step), total=args.train_step):
                try:
                    (images, labels) = next(data_iter)
                except:
                    data_iter = iter(data_loader['train_dataset'])
                    (images, labels) = next(data_iter)

                images = images.to(device, non_blocking=True)
                labels = [label.to(device, non_blocking=True) for label in labels]
                
                # Add Sample-wise Noise to each sample
                with torch.no_grad():
                    noise = model_s(images, labels, binary = False)
                    images = images + noise

                model.train()
                for param in model.parameters():
                    param.requires_grad = True
                if args.attack_type == 'min-min-max':
                    trainer.adv_train_batch(images, labels, model, optimizer, transform = transform)
                else:    
                    trainer.train_batch(images, labels, model, optimizer, transform = transform)

        # Eval termination conditions
        loss_avg, error_rate = samplewise_perturbation_eval(criterion, model_s, data_loader, model, eval_target='train_dataset', transform = transform)
        logger.info('Eval Loss: {:.4f} Acc: {:.2f}%'.format(loss_avg, 100 - error_rate*100))

        generation_steps += 1
        condition = error_rate > args.universal_stop_error
        condition = condition and (generation_steps < args.stop_epoch)

        if args.attack_type == 'min-min-max':
            save_freq = 10
        else:
            save_freq = 25
        if generation_steps % save_freq == 0:
            target_model = model_s.module if args.data_parallel else model_s
            state = {'model_state_dict': target_model.state_dict()}
            torch.save(state, os.path.join(args.exp_name, 'model_s.pth'))
            perturbations = []
            for i, (images, labels) in tqdm(enumerate(data_loader_noise), total=len(data_loader_noise)):
                images = images.to(device, non_blocking=True)
                labels = [label.to(device, non_blocking=True) for label in labels]
                with torch.no_grad():
                    noise = model_s(images, labels, binary = False)
                perturbations.append(noise.cpu().detach().clone())
            perturbations = torch.cat(perturbations, dim = 0)
            pt_name = 'perturbation.pt'
            torch.save(perturbations, os.path.join(args.exp_name, pt_name))


    # Save Model_s
    target_model = model_s.module if args.data_parallel else model_s
    state = {'model_state_dict': target_model.state_dict()}
    torch.save(state, os.path.join(args.exp_name, 'model_s.pth'))

    perturbations = []
    for i, (images, labels) in tqdm(enumerate(data_loader_noise), total=len(data_loader_noise)):
        images = images.to(device, non_blocking=True)
        labels = [label.to(device, non_blocking=True) for label in labels]
        with torch.no_grad():
            noise = model_s(images, labels, binary = False)
        perturbations.append(noise.cpu().detach().clone())
   
    perturbations = torch.cat(perturbations, dim = 0)
    return perturbations

def sample_wise_perturbation_sep(trainer, evaluator, model, criterion, optimizer, scheduler, model_s, optimizer_s, ENV):
    datasets_generator = dataset.DatasetGenerator(train_batch_size=args.train_batch_size,
                                                  eval_batch_size=args.eval_batch_size,
                                                  train_data_type=args.train_data_type,
                                                  train_data_path=args.train_data_path,
                                                  test_data_type=args.test_data_type,
                                                  test_data_path=args.test_data_path,
                                                  num_of_workers=args.num_of_workers,
                                                  seed=args.seed, no_train_augments=False, identity = args.identity)
    transform = datasets_generator.datasets['train_dataset'].transform
    transform.transforms.pop()
    print (transform)
    datasets_generator.datasets['train_dataset'].w_transform = False
    if args.train_data_type == 'ImageNetMini' and args.perturb_type == 'samplewise':
        data_loader = datasets_generator._split_validation_set(0.2, train_shuffle=False, train_drop_last=False)
        data_loader['train_dataset'] = data_loader['train_subset']
    else:
        data_loader = datasets_generator.getDataLoader(train_shuffle=True, train_drop_last=False)
    
    from torch.utils.data import DataLoader
    datasets_generator_noise = dataset.DatasetGenerator(train_batch_size=args.train_batch_size,
                                                  eval_batch_size=args.eval_batch_size,
                                                  train_data_type=args.train_data_type,
                                                  train_data_path=args.train_data_path,
                                                  test_data_type=args.test_data_type,
                                                  test_data_path=args.test_data_path,
                                                  num_of_workers=args.num_of_workers,
                                                  seed=args.seed, no_train_augments=False, identity = args.identity)
    datasets_generator_noise.datasets['train_dataset'].w_transform = False
    data_loader_noise = DataLoader(dataset=datasets_generator_noise.datasets['train_dataset'],
                                   batch_size=datasets_generator.train_batch_size,
                                   shuffle=False, pin_memory=True,
                                   drop_last=False, num_workers=datasets_generator.num_of_workers)

    condition = True
    data_iter = iter(data_loader['train_dataset'])
    logger.info('=' * 20 + 'Optimizating Samplewise Perturbator' + '=' * 20)
    generation_steps = 0
    checkpoint_path_file = args.load_model_path + '.pth'
    checkpoints = dict()
    for ckpt_epoch in range(0, 60, 4):
        checkpoints[ckpt_epoch] = torch.load(checkpoint_path_file.replace('.pth', '_epoch{}.pth'.format(ckpt_epoch)), map_location=device)['model_state_dict']

    while condition:
        # Search For Noise
        model_s.train()
        idx = 0
        for i, (images, labels) in enumerate(data_loader['train_dataset']):
            model = model.to(device)
            images = images.to(device, non_blocking=True)
            labels = [label.to(device, non_blocking=True) for label in labels]
            y = [(label + model.task_outputs[i]//2) % model.task_outputs[i] for i, label in enumerate(labels)]
            model_s.zero_grad()
            optimizer_s.zero_grad()
            for ckpt_epoch in range(0, 60, 4):
                noise = model_s(images, labels, binary = False)
                perturb_img = images + noise
                model.load_state_dict(checkpoints[ckpt_epoch], strict = True)
                model.eval()
                for param in model.parameters():
                    param.requires_grad = False
                logits = model(perturb_img)
                loss = criterion(logits, y)
                loss_optimization = loss
                if args.embedding_regularization:
                    loss_ER_weight = 10
                    loss_ER = model_s.embedding_cosine_similarity_loss()
                    loss_optimization = loss_optimization + loss_ER_weight * loss_ER
                if args.cross_embedding_regularization:
                    loss_CER_weight = 200
                    loss_CER = model_s.cross_embedding_cosine_similarity_loss()
                    loss_optimization = loss_optimization + loss_CER_weight * loss_CER
                loss_optimization.backward()
            optimizer_s.step()

            if i % 10 == 0:
                acc_metric = MultiAccuracy()
                acc_list = acc_metric(logits[:args.n_tasks], labels[:args.n_tasks])
                acc_all = sum(acc_list)/len(acc_list)
                output_info = 'Loss: {:.4f} '.format(loss)
                if args.embedding_regularization:
                    output_info = output_info + 'loss_ER: {:.4f} '.format(loss_ER)
                if args.cross_embedding_regularization:
                    output_info = output_info + 'Loss_CER: {:.4f} '.format(loss_CER)
                output_info = output_info + 'Acc_all: {:.2f}%'.format(acc_all*100)
                for i, acc in enumerate(acc_list):
                    output_info = output_info + ' Acc_{}: {:.2f}%'.format(i, acc*100)
                logger.info(output_info)
        
        scheduler_s.step()
        scheduler_s.step()
        scheduler_s.step()
        scheduler_s.step()
        model_s.eval()
        # Eval termination conditions
        loss_avg, error_rate = samplewise_perturbation_eval(criterion, model_s, data_loader, model, eval_target='train_dataset', transform = transform)
        logger.info('Eval Loss: {:.4f} Acc: {:.2f}%'.format(loss_avg, 100 - error_rate*100))

        generation_steps += 1
        condition = error_rate > args.universal_stop_error
        condition = condition and (generation_steps < args.stop_epoch//4)

        if generation_steps % 20 == 0:
            target_model = model_s.module if args.data_parallel else model_s
            state = {'model_state_dict': target_model.state_dict()}
            torch.save(state, os.path.join(args.exp_name, 'model_s.pth'))
            perturbations = []
            for i, (images, labels) in tqdm(enumerate(data_loader_noise), total=len(data_loader_noise)):
                images = images.to(device, non_blocking=True)
                labels = [label.to(device, non_blocking=True) for label in labels]
                with torch.no_grad():
                    noise = model_s(images, labels, binary = False)
                perturbations.append(noise.cpu().detach().clone())
            perturbations = torch.cat(perturbations, dim = 0)
            pt_name = 'perturbation.pt'
            torch.save(perturbations, os.path.join(args.exp_name, pt_name))


    # Save Model_s
    target_model = model_s.module if args.data_parallel else model_s
    state = {'model_state_dict': target_model.state_dict()}
    torch.save(state, os.path.join(args.exp_name, 'model_s.pth'))

    perturbations = []
    for i, (images, labels) in tqdm(enumerate(data_loader_noise), total=len(data_loader_noise)):
        images = images.to(device, non_blocking=True)
        labels = [label.to(device, non_blocking=True) for label in labels]
        with torch.no_grad():
            noise = model_s(images, labels, binary = False)
        perturbations.append(noise.cpu().detach().clone())
   
    perturbations = torch.cat(perturbations, dim = 0)
    return perturbations


def sample_wise_perturbation_mix(trainer, evaluator, model, criterion, optimizer, model_pretrained, model_s, optimizer_s, ENV):
    datasets_generator = dataset.DatasetGenerator(train_batch_size=args.train_batch_size,
                                                  eval_batch_size=args.eval_batch_size,
                                                  train_data_type=args.train_data_type,
                                                  train_data_path=args.train_data_path,
                                                  test_data_type=args.test_data_type,
                                                  test_data_path=args.test_data_path,
                                                  num_of_workers=args.num_of_workers,
                                                  seed=args.seed, no_train_augments=False, identity = args.identity)
    transform = datasets_generator.datasets['train_dataset'].transform
    transform.transforms.pop()
    print (transform)
    datasets_generator.datasets['train_dataset'].w_transform = False
    if args.train_data_type == 'ImageNetMini' and args.perturb_type == 'samplewise':
        data_loader = datasets_generator._split_validation_set(0.2, train_shuffle=False, train_drop_last=False)
        data_loader['train_dataset'] = data_loader['train_subset']
    else:
        data_loader = datasets_generator.getDataLoader(train_shuffle=True, train_drop_last=False)
    
    from torch.utils.data import DataLoader
    datasets_generator_noise = dataset.DatasetGenerator(train_batch_size=args.train_batch_size,
                                                  eval_batch_size=args.eval_batch_size,
                                                  train_data_type=args.train_data_type,
                                                  train_data_path=args.train_data_path,
                                                  test_data_type=args.test_data_type,
                                                  test_data_path=args.test_data_path,
                                                  num_of_workers=args.num_of_workers,
                                                  seed=args.seed, no_train_augments=False, identity = args.identity)
    datasets_generator_noise.datasets['train_dataset'].w_transform = False
    data_loader_noise = DataLoader(dataset=datasets_generator_noise.datasets['train_dataset'],
                                   batch_size=datasets_generator.train_batch_size,
                                   shuffle=False, pin_memory=True,
                                   drop_last=False, num_workers=datasets_generator.num_of_workers)

    condition = True
    data_iter = iter(data_loader['train_dataset'])
    logger.info('=' * 20 + 'Optimizating Samplewise Perturbator' + '=' * 20)
    generation_steps = 0
    while condition:
        logger.info('Generation_steps: {}'.format(generation_steps))
        # Search For Noise
        model_s.train()
        idx = 0
        for i, (images, labels) in enumerate(data_loader['train_dataset']):
            model = imodel.to(device)
            images = images.to(device, non_blocking=True)
            labels = [label.to(device, non_blocking=True) for label in labels]

            # Update sample-wise perturbation
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            
            model_s.zero_grad()
            optimizer_s.zero_grad()
            noise = model_s(images, labels, binary = False)
            perturb_img = images + noise
            if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                if transform is not None:
                    augmented_perturb_img = transform(perturb_img)
                    logits_min = model(augmented_perturb_img)
                    logits_max = model_pretrained(augmented_perturb_img)
                else:
                    logits_min = model(perturb_img)
                    logits_max = model_pretrained(perturb_img)
                loss_min = criterion(logits_min, labels)
                loss_max = criterion(logits_max, labels)
                
            loss_optimization = loss_min - loss_max
            loss_optimization.backward()
            optimizer_s.step()

            if i % 10 == 0:
                acc_min = ((logits_min > 0).eq(labels).float())[:,:args.n_tasks].mean().cpu().item()
                acc_max = ((logits_max > 0).eq(labels).float())[:,:args.n_tasks].mean().cpu().item()
                logger.info('Loss: {:.4f} Loss_min: {:.4f} Loss_max: {:.4f} Acc_min: {:.2f} Acc_max: {:.2f}%'.format(loss_optimization, loss_min, loss_max, acc_min*100, acc_max*100))
        
        model_s.eval()
        # Train Batch for min-min noise
        for j in tqdm(range(0, args.train_step), total=args.train_step):
            try:
                (images, labels) = next(data_iter)
            except:
                data_iter = iter(data_loader['train_dataset'])
                (images, labels) = next(data_iter)

            images = images.to(device, non_blocking=True)
            labels = [label.to(device, non_blocking=True) for label in labels]
            
            # Add Sample-wise Noise to each sample
            with torch.no_grad():
                noise = model_s(images, labels, binary = False)
                images = images + noise

            model.train()
            for param in model.parameters():
                param.requires_grad = True
            trainer.train_batch(images, labels, model, optimizer, transform = transform)

        # Eval termination conditions
        loss_avg, error_rate = samplewise_perturbation_eval(model_s, data_loader, model, eval_target='train_dataset', transform = transform)
        logger.info('Eval Loss: {:.4f} Acc: {:.2f}%'.format(loss_avg, 100 - error_rate*100))

        generation_steps += 1
        condition = error_rate > args.universal_stop_error

        if generation_steps % 50 == 0:
            target_model = model_s.module if args.data_parallel else model_s
            state = {'model_state_dict': target_model.state_dict()}
            torch.save(state, os.path.join(args.exp_name, 'model_s.pth'))
            perturbations = []
            for i, (images, labels) in tqdm(enumerate(data_loader_noise), total=len(data_loader_noise)):
                images = images.to(device, non_blocking=True)
                labels = [label.to(device, non_blocking=True) for label in labels]
                with torch.no_grad():
                    noise = model_s(images, labels, binary = False)
                perturbations.append(noise.cpu().detach().clone())
            perturbations = torch.cat(perturbations, dim = 0)
            pt_name = 'perturbation.pt'
            torch.save(perturbations, os.path.join(args.exp_name, pt_name))


    # Save Model_s
    target_model = model_s.module if args.data_parallel else model_s
    state = {'model_state_dict': target_model.state_dict()}
    torch.save(state, os.path.join(args.exp_name, 'model_s.pth'))

    perturbations = []
    for i, (images, labels) in tqdm(enumerate(data_loader_noise), total=len(data_loader_noise)):
        images = images.to(device, non_blocking=True)
        labels = [label.to(device, non_blocking=True) for label in labels]
        with torch.no_grad():
            noise = model_s(images, labels, binary = False)
        perturbations.append(noise.cpu().detach().clone())
   
    perturbations = torch.cat(perturbations, dim = 0)
    return perturbations

def main():
    if not ('min-min' in args.attack_type):
        # Setup ENV
        datasets_generator = dataset.DatasetGenerator(train_batch_size=args.train_batch_size,
                                                      eval_batch_size=args.eval_batch_size,
                                                      train_data_type=args.train_data_type,
                                                      train_data_path=args.train_data_path,
                                                      test_data_type=args.test_data_type,
                                                      test_data_path=args.test_data_path,
                                                      num_of_workers=args.num_of_workers,
                                                      seed=args.seed, identity = args.identity)
        data_loader = datasets_generator.getDataLoader()
    else:
        datasets_generator = None
        data_loader = None

    model = config.model().to(device)
    logger.info("param size = %fMB", util.count_parameters_in_MB(model))
    optimizer = config.optimizer([dict(params=model.shared_parameters()), dict(params=model.task_specific_parameters())])
    scheduler = config.scheduler(optimizer)
    criterion = config.criterion()

    config_s['model']['attr_num'] = args.n_tasks
    model_s = config_s.model().to(device)
    optimizer_s = config_s.optimizer(model_s.parameters())
    global scheduler_s
    scheduler_s = config_s.scheduler(optimizer_s)

    global loss_weight
    loss_weight = torch.zeros(model.n_tasks).float().cuda()
    loss_weight[:args.n_tasks] = 1
    criterion.weight = loss_weight
    if 'pos_weight' in config:
        print ('use_balanced_weight')
        criterion.pos_weight = torch.tensor(config['pos_weight']).to(device)
    if args.perturb_type == 'samplewise':
        train_target = 'train_dataset'
    else:
        if args.use_subset:
            data_loader = datasets_generator._split_validation_set(train_portion=args.universal_train_portion,
                                                                   train_shuffle=True, train_drop_last=True)
            train_target = 'train_subset'
        else:
            data_loader = datasets_generator.getDataLoader(train_shuffle=True, train_drop_last=True)
            train_target = 'train_dataset'

    trainer = Trainer(criterion, data_loader, logger, config, target=train_target)
    evaluator = Evaluator(criterion, data_loader, logger, config)
    ENV = {'global_step': 0,
           'best_acc': 0.0,
           'curren_acc': 0.0,
           'best_pgd_acc': 0.0,
           'train_history': [],
           'eval_history': [],
           'pgd_eval_history': [],
           'genotype_list': []}

    if args.data_parallel:
        model = torch.nn.DataParallel(model)
        model_s = torch.nn.DataParallel(model_s)

    if args.attack_type in ['mix']:
        import copy
        model_surrogate = copy.deepcopy(model)
        optimizer_surrogate = config.optimizer([dict(params=model_surrogate.shared_parameters()), dict(params=model_surrogate.task_specific_parameters())])
        scheduler_surrogate = config.scheduler(optimizer_surrogate)

    if args.load_model:
        checkpoint_path_file = args.load_model_path
        checkpoint = util.load_model(filename=checkpoint_path_file,
                                     model=model,
                                     optimizer=optimizer,
                                     alpha_optimizer=None,
                                     scheduler=scheduler)
        ENV = checkpoint['ENV']
        trainer.global_step = ENV['global_step']
        logger.info("File %s loaded!" % (checkpoint_path_file))

    if args.attack_type in ['min-min', 'min-min-max', 'min-max']:
        if args.attack_type in ['min-max'] and (not args.load_model):
            # min-max noise need model to converge first
            train(0, model, optimizer, scheduler, criterion, trainer, evaluator, ENV, data_loader)
        if not ('min-min' in args.attack_type):
            del data_loader
        if args.perturb_type == 'samplewise':
            noise = sample_wise_perturbation(trainer, evaluator, model, criterion, optimizer, scheduler, model_s, optimizer_s, ENV)
        elif args.perturb_type == 'classwise':
            noise = universal_perturbation(noise_generator, trainer, evaluator, model, criterion, optimizer, scheduler, model_s, ENV)
    elif args.attack_type in ['mix']:
        if not args.load_model:
            train(0, model, optimizer, scheduler, criterion, trainer, evaluator, ENV, data_loader)
        del data_loader
        if args.perturb_type == 'samplewise':
            noise = sample_wise_perturbation_mix(trainer, evaluator, model_surrogate, criterion, optimizer_surrogate, model, model_s, optimizer_s, ENV)
    elif args.attack_type == 'sep':
        del data_loader
        noise = sample_wise_perturbation_sep(trainer, evaluator, model, criterion, optimizer, scheduler, model_s, optimizer_s, ENV)
    elif args.attack_type == 'sep_fa':
        raise NotImplementedError("This method is not yet implemented.")
        del data_loader
        noise = sample_wise_perturbation_sep_fa(trainer, evaluator, model, criterion, optimizer, scheduler, model_s, optimizer_s, ENV)
    else:
        raise('Not implemented yet')    
    
    pt_name = 'perturbation.pt'
    torch.save(noise, os.path.join(args.exp_name, pt_name))
    logger.info(noise)
    logger.info(noise.shape)
    logger.info('Noise saved at %s' % (os.path.join(args.exp_name, pt_name)))
    return


if __name__ == '__main__':
    for arg in vars(args):
        logger.info("%s: %s" % (arg, getattr(args, arg)))
    start = time.time()
    main()
    end = time.time()
    cost = (end - start) / 86400
    payload = "Running Cost %.2f Days \n" % cost
    logger.info(payload)
