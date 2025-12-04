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

class Evaluator():
    def __init__(self, criterion, data_loader, logger, config):
        self.loss_meters = util.AverageMeter()
        self.acc_all_meters = util.AverageMeter()
        self.acc_meters = dict()
        for i in range(args.n_tasks):
            self.acc_meters[i] = util.AverageMeter()
        if hasattr(criterion, 'cross_entropy'):
            self.criterion = criterion.cross_entropy
        else:    
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

    def train_batch(self, images, labels, model, optimizer):
        model.zero_grad()
        if hasattr(self.criterion, 'cross_entropy'):
            logits, loss = self.criterion(model, images, labels, optimizer)
        else:
            optimizer.zero_grad()
            logits = model(images)
            if args.regularization:
                lambda_reg = 100
                loss = self.criterion(logits, labels) + lambda_reg * model.regularization_loss()
            else:
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
parser.add_argument('--num_of_workers', default=8, type=int, help='workers for loader')
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

parser.add_argument('--n_tasks', default=3, type=int)
parser.add_argument('--regularization', action='store_true', default=False)

parser.add_argument('--partial_rate', default=1.0, type=float)

parser.add_argument('--pretrained_encoder', action='store_true', default=False)
parser.add_argument('--pretrained_encoder_path', default='pretrained/resnet18.pth', type=str)
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
        # ENV['train_history'].append(trainer.acc_all_meters.avg*100)
        scheduler.step()

        # Eval
        logger.info("="*20 + "Eval Epoch %d" % (epoch) + "="*20)
        is_best = False
        if not args.train_face:
            evaluator.eval(epoch, model)
            payload = 'Eval Loss:%.4f\tEval acc: %.2f' % (evaluator.loss_meters.avg, evaluator.acc_all_meters.avg*100)
            for i in range(args.n_tasks):
                payload = payload + '\tEval acc_%d: %.2f' % (i, evaluator.acc_meters[i].avg*100)
            payload = (payload)
            # payload = ('Eval Loss:%.4f\tEval acc: %.2f' % (evaluator.loss_meters.avg, evaluator.acc_all_meters.avg*100))
            logger.info(payload)
            # ENV['eval_history'].append(evaluator.acc_all_meters.avg*100)
            # ENV['curren_acc'] = evaluator.acc_all_meters.avg*100
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
    evaluator = Evaluator(criterion, data_loader, logger, config)

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


    if args.pretrained_encoder:
        print ("Using pretrained encoder!")
        state_dict = torch.load(args.pretrained_encoder_path)
        model.load_state_dict(state_dict, strict = False)

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
        evaluator.eval(starting_epoch, model)
        payload = ('Eval Loss:%.4f\tEval acc: %.2f' % (evaluator.loss_meters.avg, evaluator.acc_all_meters.avg*100))
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
