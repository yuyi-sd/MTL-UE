import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import util

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class PerturbationTool():
    def __init__(self, seed=0, epsilon=0.03137254901, num_steps=20, step_size=0.00784313725):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.seed = seed
        np.random.seed(seed)

    def random_noise(self, noise_shape=[10, 3, 32, 32]):
        random_noise = torch.FloatTensor(*noise_shape).uniform_(-self.epsilon, self.epsilon).to(device)
        return random_noise

    def min_min_attack(self, images, labels, model, optimizer, criterion, random_noise=None, sample_wise=False, transform = None):
        if random_noise is None:
            random_noise = torch.FloatTensor(*images.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb_img = Variable(images.data + random_noise, requires_grad=True)
        perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)
        eta = random_noise
        for _ in range(self.num_steps):
            opt = torch.optim.SGD([perturb_img], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                if hasattr(model, 'classify'):
                    model.classify = True
                if transform is not None:
                    logits = model(transform(perturb_img))
                else:
                    logits = model(perturb_img)
                loss = criterion(logits, labels)
            elif isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                if transform is not None:
                    logits = model(transform(perturb_img))
                else:
                    logits = model(perturb_img)
                loss = criterion(logits, labels)
            elif 'MultiCrossEntropyLoss' in str(type(criterion)):
                if transform is not None:
                    logits = model(transform(perturb_img))
                else:
                    logits = model(perturb_img)
                loss = criterion(logits, labels)
            perturb_img.retain_grad()
            loss.backward()
            eta = self.step_size * perturb_img.grad.data.sign() * (-1)
            perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
            eta = torch.clamp(perturb_img.data - images.data, -self.epsilon, self.epsilon)
            perturb_img = Variable(images.data + eta, requires_grad=True)
            perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)

        return perturb_img, eta

    def min_min_max_attack(self, images, labels, model, optimizer, criterion, random_noise=None, sample_wise=False, transform = None):
        if random_noise is None:
            random_noise = torch.FloatTensor(*images.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb_img = Variable(images.data + random_noise, requires_grad=True)
        perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)
        eta = random_noise
        for _ in range(self.num_steps):
            opt = torch.optim.SGD([perturb_img], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()
            for _ in range(self.num_steps):
                random_noise_max = torch.FloatTensor(*images.shape).uniform_(-self.epsilon/2, self.epsilon/2).to(device).requires_grad_()
                opt2 = torch.optim.SGD([random_noise_max], lr=1e-3)
                opt2.zero_grad()
                model.zero_grad()
                if isinstance(criterion, torch.nn.CrossEntropyLoss):
                    if hasattr(model, 'classify'):
                        model.classify = True
                    if transform is not None:
                        logits = model(transform(perturb_img)+random_noise_max)
                    else:
                        logits = model(perturb_img+random_noise_max)
                    loss = criterion(logits, labels)
                elif isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                    if transform is not None:
                        logits = model(transform(perturb_img)+random_noise_max)
                    else:
                        logits = model(perturb_img+random_noise_max)
                    loss = criterion(logits, labels)
                elif 'MultiCrossEntropyLoss' in str(type(criterion)):
                    if transform is not None:
                        logits = model(transform(perturb_img)+random_noise_max)
                    else:
                        logits = model(perturb_img+random_noise_max)
                    loss = criterion(logits, labels)
                random_noise_max.retain_grad()
                loss.backward()
                eta = self.step_size/2 * random_noise_max.grad.data.sign()
                random_noise_max = Variable(random_noise_max.data + eta, requires_grad=True)
                random_noise_max = Variable(torch.clamp(random_noise_max, -self.epsilon/2, self.epsilon/2), requires_grad=True)
            random_noise_max = random_noise_max.clone().detach()
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                if hasattr(model, 'classify'):
                    model.classify = True
                if transform is not None:
                    logits = model(transform(perturb_img)+random_noise_max)
                else:
                    logits = model(perturb_img+random_noise_max)
                loss = criterion(logits, labels)
            elif isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                if transform is not None:
                    logits = model(transform(perturb_img)+random_noise_max)
                else:
                    logits = model(perturb_img+random_noise_max)
                loss = criterion(logits, labels)
            elif 'MultiCrossEntropyLoss' in str(type(criterion)):
                if transform is not None:
                    logits = model(transform(perturb_img)+random_noise_max)
                else:
                    logits = model(perturb_img+random_noise_max)
                loss = criterion(logits, labels)
            perturb_img.retain_grad()
            loss.backward()
            eta = self.step_size * perturb_img.grad.data.sign() * (-1)
            perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
            eta = torch.clamp(perturb_img.data - images.data, -self.epsilon, self.epsilon)
            perturb_img = Variable(images.data + eta, requires_grad=True)
            perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)

        return perturb_img, eta

    def min_max_attack(self, images, labels, model, optimizer, criterion, random_noise=None, sample_wise=False):
        if random_noise is None:
            random_noise = torch.FloatTensor(*images.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb_img = Variable(images.data + random_noise, requires_grad=True)
        perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)
        eta = random_noise
        for _ in range(self.num_steps):
            opt = torch.optim.SGD([perturb_img], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                logits = model(perturb_img)
                loss = criterion(logits, labels)
            elif isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                logits = model(perturb_img)
                loss = criterion(logits, labels)
            elif 'MultiCrossEntropyLoss' in str(type(criterion)):
                # logits, loss = criterion(model, perturb_img, labels, optimizer)
                logits = model(perturb_img)
                y = [(label + model.task_outputs[i]//2) % model.task_outputs[i] for i, label in enumerate(labels)]
                loss = - criterion(logits, y)
            loss.backward()

            eta = self.step_size * perturb_img.grad.data.sign()
            perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
            eta = torch.clamp(perturb_img.data - images.data, -self.epsilon, self.epsilon)
            perturb_img = Variable(images.data + eta, requires_grad=True)
            perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)

        return perturb_img, eta

    def sep_binary(self, images, labels, model, criterion, checkpoint_path_file, Features_mean, n_tasks, random_noise=None, sample_wise=False):
        if random_noise is None:
            random_noise = torch.FloatTensor(*images.shape).uniform_(-self.epsilon, self.epsilon).to(device)
        perturb_img = Variable(images.data + random_noise, requires_grad=True)
        perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)
        eta = random_noise
        
        if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
            y = (labels + 1) % 2 # B * N
        elif 'MultiCrossEntropyLoss' in str(type(criterion)):
            y = [(label + model.task_outputs[i]//2) % model.task_outputs[i] for i, label in enumerate(labels)]
        for _ in range(self.num_steps):
            grads = torch.tensor([]).to(device)
            for ckpt_epoch in range(0, 60, 4):
                checkpoint = util.load_model(filename=checkpoint_path_file.replace('.pth', '_epoch{}.pth'.format(ckpt_epoch)),
                                         model=model,
                                         optimizer=None,
                                         alpha_optimizer=None,
                                         scheduler=None)
                model.eval()
                opt = torch.optim.SGD([perturb_img], lr=1e-3)
                opt.zero_grad()
                model.zero_grad()
                logits = model(perturb_img)
                loss = criterion(logits, y)
                loss.backward()

                grads = torch.cat([grads, perturb_img.grad.data.clone().detach().unsqueeze(0)], dim=0)

            grad_avg = grads.mean(0)
            grad_sum = grad_avg
            eta = self.step_size * grad_sum.sign()
            perturb_img = Variable(perturb_img.data - eta, requires_grad=True)
            eta = torch.clamp(perturb_img.data - images.data, -self.epsilon, self.epsilon)
            perturb_img = Variable(images.data + eta, requires_grad=True)
            perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)

        return perturb_img, eta

    def sep_fa_binary(self, images, labels, model, checkpoint_path_file, Features_mean, n_tasks, random_noise=None, sample_wise=False):
        if random_noise is None:
            random_noise = torch.FloatTensor(*images.shape).uniform_(-self.epsilon, self.epsilon).to(device)
        criterion = nn.MSELoss()
        perturb_img = Variable(images.data + random_noise, requires_grad=True)
        perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)
        eta = random_noise
        
        if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
            y = (labels + 1) % 2 # B * N
        elif 'MultiCrossEntropyLoss' in str(type(criterion)):
            raise NotImplementedError("This method is not yet implemented.")
            y = [(label + model.task_outputs[i]//2) % model.task_outputs[i] for i, label in enumerate(labels)]
        # print (Features_mean.shape)
        # print (selected_Features_mean.shape)
        # print (y.shape)
        for _ in range(self.num_steps):
            grads = torch.tensor([]).to(device)
            for ckpt_epoch in range(0, 60, 4):
                selected_Features_mean = Features_mean[ckpt_epoch].unsqueeze(0).repeat(perturb_img.shape[0], 1, 1, 1) # B * N * 2 * ...
                selected_Features_mean = selected_Features_mean[torch.arange(perturb_img.shape[0]).unsqueeze(1), torch.arange(n_tasks), y, ...] # B * N * ...
                checkpoint = util.load_model(filename=checkpoint_path_file.replace('.pth', '_epoch{}.pth'.format(ckpt_epoch)),
                                         model=model,
                                         optimizer=None,
                                         alpha_optimizer=None,
                                         scheduler=None)
                model.eval()
                opt = torch.optim.SGD([perturb_img], lr=1e-3)
                opt.zero_grad()
                model.zero_grad()
                features = model(perturb_img, return_representation=True)[1] # B * ...
                features = features.unsqueeze(1).repeat(1, n_tasks, 1) # B * N * ...
                loss = criterion(features, selected_Features_mean)
                loss.backward()

                grads = torch.cat([grads, perturb_img.grad.data.clone().detach().unsqueeze(0)], dim=0)

            grad_avg = grads.mean(0)

            grad_sum = grad_avg
            eta = self.step_size * grad_sum.sign()
            perturb_img = Variable(perturb_img.data - eta, requires_grad=True)
            eta = torch.clamp(perturb_img.data - images.data, -self.epsilon, self.epsilon)
            perturb_img = Variable(images.data + eta, requires_grad=True)
            perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)

        return perturb_img, eta

    def sep_fa_vr_binary(self, images, labels, model, checkpoint_path_file, Features_mean, n_tasks, random_noise=None, sample_wise=False):
        if random_noise is None:
            random_noise = torch.FloatTensor(*images.shape).uniform_(-self.epsilon, self.epsilon).to(device)
        criterion = nn.MSELoss()
        perturb_img = Variable(images.data + random_noise, requires_grad=True)
        perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)
        eta = random_noise
        
        if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
            y = (labels + 1) % 2 # B * N
        elif 'MultiCrossEntropyLoss' in str(type(criterion)):
            raise NotImplementedError("This method is not yet implemented.")
            y = [(label + model.task_outputs[i]//2) % model.task_outputs[i] for i, label in enumerate(labels)]
        # print (Features_mean.shape)
        # print (selected_Features_mean.shape)
        # print (y.shape)
        for _ in range(self.num_steps):
            grads = torch.tensor([]).to(device)
            for ckpt_epoch in range(0, 60, 4):
                selected_Features_mean = Features_mean[ckpt_epoch].unsqueeze(0).repeat(perturb_img.shape[0], 1, 1, 1) # B * N * 2 * ...
                selected_Features_mean = selected_Features_mean[torch.arange(perturb_img.shape[0]).unsqueeze(1), torch.arange(n_tasks), y, ...] # B * N * ...
                checkpoint = util.load_model(filename=checkpoint_path_file.replace('.pth', '_epoch{}.pth'.format(ckpt_epoch)),
                                         model=model,
                                         optimizer=None,
                                         alpha_optimizer=None,
                                         scheduler=None)
                model.eval()
                opt = torch.optim.SGD([perturb_img], lr=1e-3)
                opt.zero_grad()
                model.zero_grad()
                features = model(perturb_img, return_representation=True)[1] # B * ...
                features = features.unsqueeze(1).repeat(1, n_tasks, 1) # B * N * ...
                loss = criterion(features, selected_Features_mean)
                loss.backward()

                grads = torch.cat([grads, perturb_img.grad.data.clone().detach().unsqueeze(0)], dim=0)

            grad_avg = grads.mean(0)

            grad_sum = 0
            perturb_img_v = Variable(perturb_img.detach().clone().data, requires_grad=True)
            for k in range(15):
                ind_ckpt = np.random.choice(range(15))
                ckpt_epoch = list(range(0, 60, 4))[ind_ckpt]
                selected_Features_mean = Features_mean[ckpt_epoch].unsqueeze(0).repeat(perturb_img.shape[0], 1, 1, 1) # B * N * 2 * ...
                selected_Features_mean = selected_Features_mean[torch.arange(perturb_img.shape[0]).unsqueeze(1), torch.arange(n_tasks), y, ...] # B * N * ...
                checkpoint = util.load_model(filename=checkpoint_path_file.replace('.pth', '_epoch{}.pth'.format(ckpt_epoch)),
                                         model=model,
                                         optimizer=None,
                                         alpha_optimizer=None,
                                         scheduler=None)
                model.eval()
                opt = torch.optim.SGD([perturb_img_v], lr=1e-3)
                opt.zero_grad()
                model.zero_grad()
                features = model(perturb_img_v, return_representation=True)[1] # B * ...
                features = features.unsqueeze(1).repeat(1, n_tasks, 1) # B * N * ...
                loss = criterion(features, selected_Features_mean)
                loss.backward()

                grad_v = perturb_img_v.grad.data
                grad_sum += grad_v - (grads[ind_ckpt] - grad_avg)

                eta = self.step_size * grad_sum.sign()
                perturb_img_v = Variable(perturb_img_v.data - eta, requires_grad=True)
                eta = torch.clamp(perturb_img_v.data - images.data, -self.epsilon, self.epsilon)
                perturb_img_v = Variable(images.data + eta, requires_grad=True)
                perturb_img_v = Variable(torch.clamp(perturb_img_v, 0, 1), requires_grad=True)

            eta = self.step_size * grad_sum.sign()
            perturb_img = Variable(perturb_img.data - eta, requires_grad=True)
            eta = torch.clamp(perturb_img.data - images.data, -self.epsilon, self.epsilon)
            perturb_img = Variable(images.data + eta, requires_grad=True)
            perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)

        return perturb_img, eta

    def _patch_noise_extend_to_img(self, noise, image_size=[3, 32, 32], patch_location='center'):
        c, h, w = image_size[0], image_size[1], image_size[2]
        mask = np.zeros((c, h, w), np.float32)
        x_len, y_len = noise.shape[1], noise.shape[1]

        if patch_location == 'center' or (h == w == x_len == y_len):
            x = h // 2
            y = w // 2
        elif patch_location == 'random':
            x = np.random.randint(x_len // 2, w - x_len // 2)
            y = np.random.randint(y_len // 2, h - y_len // 2)
        else:
            raise('Invalid patch location')

        x1 = np.clip(x - x_len // 2, 0, h)
        x2 = np.clip(x + x_len // 2, 0, h)
        y1 = np.clip(y - y_len // 2, 0, w)
        y2 = np.clip(y + y_len // 2, 0, w)
        if type(noise) is np.ndarray:
            pass
        else:
            mask[:, x1: x2, y1: y2] = noise.cpu().numpy()
        return ((x1, x2, y1, y2), torch.from_numpy(mask).to(device))






class PerturbationTool_MTL():
    def __init__(self, seed=0, epsilon=0.03137254901, num_steps=20, step_size=0.00784313725):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.seed = seed
        np.random.seed(seed)

    def random_noise(self, noise_shape=[10, 3, 32, 32]):
        random_noise = torch.FloatTensor(*noise_shape).uniform_(-self.epsilon, self.epsilon).to(device)
        return random_noise

    def min_min_attack(self, args, images, a, g, r, model, optimizer, criterion, random_noise=None, sample_wise=False):
        if random_noise is None:
            random_noise = torch.FloatTensor(*images.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb_img = Variable(images.data + random_noise, requires_grad=True)
        perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)
        eta = random_noise
        for _ in range(self.num_steps):
            opt = torch.optim.SGD([perturb_img], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                if hasattr(model, 'classify'):
                    model.classify = True
                # logits = model(perturb_img)
                # loss = criterion(logits, labels)
                logits, features = model(perturb_img, return_representation=True)
                loss_a = criterion(logits[0], a)
                loss_g = criterion(logits[1], g)
                loss_r = criterion(logits[2], r)
                loss = 0
                if args.train_a:
                    loss += loss_a
                if args.train_g:
                    loss += loss_g
                if args.train_r:
                    loss += loss_r

            # else:
            #     logits, loss = criterion(model, perturb_img, labels, optimizer)
            perturb_img.retain_grad()
            loss.backward()
            eta = self.step_size * perturb_img.grad.data.sign() * (-1)
            perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
            eta = torch.clamp(perturb_img.data - images.data, -self.epsilon, self.epsilon)
            perturb_img = Variable(images.data + eta, requires_grad=True)
            perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)

        return perturb_img, eta

    def min_max_attack(self, images, labels, model, optimizer, criterion, random_noise=None, sample_wise=False):
        if random_noise is None:
            random_noise = torch.FloatTensor(*images.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb_img = Variable(images.data + random_noise, requires_grad=True)
        perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)
        eta = random_noise
        for _ in range(self.num_steps):
            opt = torch.optim.SGD([perturb_img], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                logits = model(perturb_img)
                loss = criterion(logits, labels)
            else:
                logits, loss = criterion(model, perturb_img, labels, optimizer)
            loss.backward()

            eta = self.step_size * perturb_img.grad.data.sign()
            perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
            eta = torch.clamp(perturb_img.data - images.data, -self.epsilon, self.epsilon)
            perturb_img = Variable(images.data + eta, requires_grad=True)
            perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)

        return perturb_img, eta

    def _patch_noise_extend_to_img(self, noise, image_size=[3, 32, 32], patch_location='center'):
        c, h, w = image_size[0], image_size[1], image_size[2]
        mask = np.zeros((c, h, w), np.float32)
        x_len, y_len = noise.shape[1], noise.shape[1]

        if patch_location == 'center' or (h == w == x_len == y_len):
            x = h // 2
            y = w // 2
        elif patch_location == 'random':
            x = np.random.randint(x_len // 2, w - x_len // 2)
            y = np.random.randint(y_len // 2, h - y_len // 2)
        else:
            raise('Invalid patch location')

        x1 = np.clip(x - x_len // 2, 0, h)
        x2 = np.clip(x + x_len // 2, 0, h)
        y1 = np.clip(y - y_len // 2, 0, w)
        y2 = np.clip(y + y_len // 2, 0, w)
        if type(noise) is np.ndarray:
            pass
        else:
            mask[:, x1: x2, y1: y2] = noise.cpu().numpy()
        return ((x1, x2, y1, y2), torch.from_numpy(mask).to(device))



class PerturbationTool_VControl():
    def __init__(self, seed=0, epsilon=0.03137254901, num_steps=20, step_size=0.00784313725):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.seed = seed
        np.random.seed(seed)

    def random_noise(self, noise_shape=[10, 3, 32, 32]):
        random_noise = torch.FloatTensor(*noise_shape).uniform_(-self.epsilon, self.epsilon).to(device)
        return random_noise

    def min_min_attack(self, images, labels, model, optimizer, criterion, random_noise=None, sample_wise=False, transform = None, vweights_avg = 1.0, vweights_max = 1.0, vcontrol = True):
        if random_noise is None:
            random_noise = torch.FloatTensor(*images.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb_img = Variable(images.data + random_noise, requires_grad=True)
        perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)
        eta = random_noise
        for _ in range(self.num_steps):
            opt = torch.optim.SGD([perturb_img], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                if hasattr(model, 'classify'):
                    model.classify = True
                if transform is not None:
                    logits = model(transform(perturb_img))
                else:
                    logits = model(perturb_img)
                loss = criterion(logits, labels)
            elif isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                if transform is not None:
                    logits, features = model(transform(perturb_img), return_representation=True)
                else:
                    logits, features = model(perturb_img, return_representation=True)
                loss_cls = criterion(logits, labels)
                if vcontrol:
                    from collections import defaultdict
                    self.feats = defaultdict(lambda: {0: [], 1: []})
                    for j in range(len(images)):
                        for ii in range(logits.shape[1]):
                            self.feats[ii][int(labels[j, ii])].append(features[j])
                    Relative_std = 0
                    max_Relative_std = 0
                    for ii in range(logits.shape[1]):
                        feats_0 = torch.stack(self.feats[ii][0]) if self.feats[ii][0] else torch.empty(0, features.shape[1], device=device)
                        feats_1 = torch.stack(self.feats[ii][1]) if self.feats[ii][1] else torch.empty(0, features.shape[1], device=device)
                        if feats_0.shape[0] > 1:
                            mean_0, variance_0 = feats_0.mean(dim=0, keepdim=True), feats_0.var(dim=0, unbiased=False, keepdim=True)
                        else:
                            mean_0, variance_0 = torch.zeros(1, features.shape[1], device=device), torch.zeros(1, features.shape[1], device=device)
                        if feats_1.shape[0] > 1:
                            mean_1, variance_1 = feats_1.mean(dim=0, keepdim=True), feats_1.var(dim=0, unbiased=False, keepdim=True)
                        else:
                            mean_1, variance_1 = torch.zeros(1, features.shape[1], device=device), torch.zeros(1, features.shape[1], device=device)
                        relative_variance_0 = variance_0 / (mean_0 ** 2 + 1e-6)
                        relative_variance_1 = variance_1 / (mean_1 ** 2 + 1e-6)
                        std_sum = (relative_variance_0.sqrt() + relative_variance_1.sqrt()).squeeze()
                        Relative_std += std_sum.mean()
                        max_Relative_std += std_sum.max()
                    Relative_std = Relative_std / 80.0
                    max_Relative_std = max_Relative_std / 80.0
                    loss = loss_cls + vweights_avg * Relative_std + vweights_max * max_Relative_std
                else:
                    loss = loss_cls
            elif 'MultiCrossEntropyLoss' in str(type(criterion)):
                if transform is not None:
                    logits, features = model(transform(perturb_img), return_representation=True)
                else:
                    logits, features = model(perturb_img, return_representation=True)
                self.feats = dict()
                for ii in range(logits.shape[1]):
                    self.feats[ii] = dict()
                    self.feats[ii][0] = torch.tensor([]).to(device)
                    self.feats[ii][1] = torch.tensor([]).to(device)
                for j in range(len(images)):
                    for ii in range(logits.shape[1]):
                        if labels[j, ii] == 0:
                            self.feats[ii][0] = torch.cat([self.feats[ii][0], features[j].unsqueeze(0)], dim=0)
                        if labels[j, ii] == 1:
                            self.feats[ii][1] = torch.cat([self.feats[ii][1], features[j].unsqueeze(0)], dim=0)
                Relative_std = 0
                max_Relative_std = 0
                for ii in range(logits.shape[1]):
                    variance_0 = self.feats[ii][0].var(dim=0, unbiased=False).squeeze()
                    variance_1 = self.feats[ii][1].var(dim=0, unbiased=False).squeeze()
                    mean_0 = self.feats[ii][0].mean(dim=0, keepdim=True)
                    mean_1 = self.feats[ii][1].mean(dim=0, keepdim=True)
                    variance_0 = self.feats[ii][0].var(dim=0, keepdim=True)
                    variance_1 = self.feats[ii][1].var(dim=0, keepdim=True)
                    relative_variance_0 = variance_0 / (mean_0 ** 2 + 1e-6)
                    relative_variance_1 = variance_1 / (mean_1 ** 2 + 1e-6)
                    Relative_std = Relative_std + (relative_variance_0 ** 0.5 + relative_variance_1 ** 0.5).squeeze().mean()
                    max_Relative_std = max_Relative_std + (relative_variance_0 ** 0.5 + relative_variance_1 ** 0.5).squeeze().max()
                loss = criterion(logits, labels) + vweights_avg * Relative_std + vweights_max * max_Relative_std
            perturb_img.retain_grad()
            loss.backward()
            eta = self.step_size * perturb_img.grad.data.sign() * (-1)
            perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
            eta = torch.clamp(perturb_img.data - images.data, -self.epsilon, self.epsilon)
            perturb_img = Variable(images.data + eta, requires_grad=True)
            perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)

        return perturb_img, eta

    def min_max_attack(self, images, labels, model, optimizer, criterion, random_noise=None, sample_wise=False, vweights_avg = 1.0, vweights_max = 1.0):
        if random_noise is None:
            random_noise = torch.FloatTensor(*images.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb_img = Variable(images.data + random_noise, requires_grad=True)
        perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)
        eta = random_noise
        for _ in range(self.num_steps):
            opt = torch.optim.SGD([perturb_img], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                logits = model(perturb_img)
                loss = criterion(logits, labels)
            elif isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                logits, features = model(perturb_img, return_representation=True)
                # self.feats = dict()
                # for ii in range(logits.shape[1]):
                #     self.feats[ii] = dict()
                #     self.feats[ii][0] = torch.tensor([]).to(device)
                #     self.feats[ii][1] = torch.tensor([]).to(device)
                # for j in range(len(images)):
                #     for ii in range(logits.shape[1]):
                #         if labels[j, ii] == 0:
                #             self.feats[ii][0] = torch.cat([self.feats[ii][0], features[j].unsqueeze(0)], dim=0)
                #         if labels[j, ii] == 1:
                #             self.feats[ii][1] = torch.cat([self.feats[ii][1], features[j].unsqueeze(0)], dim=0)
                # Relative_std = 0
                # max_Relative_std = 0
                # for ii in range(logits.shape[1]):
                #     variance_0 = self.feats[ii][0].var(dim=0, unbiased=False).squeeze()
                #     variance_1 = self.feats[ii][1].var(dim=0, unbiased=False).squeeze()
                #     mean_0 = self.feats[ii][0].mean(dim=0, keepdim=True)
                #     mean_1 = self.feats[ii][1].mean(dim=0, keepdim=True)
                #     variance_0 = self.feats[ii][0].var(dim=0, keepdim=True)
                #     variance_1 = self.feats[ii][1].var(dim=0, keepdim=True)
                #     relative_variance_0 = variance_0 / (mean_0 ** 2 + 1e-6)
                #     relative_variance_1 = variance_1 / (mean_1 ** 2 + 1e-6)
                #     Relative_std = Relative_std + (relative_variance_0 ** 0.5 + relative_variance_1 ** 0.5).squeeze().mean()
                #     max_Relative_std = max_Relative_std + (relative_variance_0 ** 0.5 + relative_variance_1 ** 0.5).squeeze().max()
                from collections import defaultdict
                self.feats = defaultdict(lambda: {0: [], 1: []})
                for j in range(len(images)):
                    for ii in range(logits.shape[1]):
                        self.feats[ii][int(labels[j, ii])].append(features[j])
                Relative_std = 0
                max_Relative_std = 0
                for ii in range(logits.shape[1]):
                    feats_0 = torch.stack(self.feats[ii][0]) if self.feats[ii][0] else torch.empty(0, features.shape[1], device=device)
                    feats_1 = torch.stack(self.feats[ii][1]) if self.feats[ii][1] else torch.empty(0, features.shape[1], device=device)
                    if feats_0.shape[0] > 1:
                        mean_0, variance_0 = feats_0.mean(dim=0, keepdim=True), feats_0.var(dim=0, unbiased=False, keepdim=True)
                    else:
                        mean_0, variance_0 = torch.zeros(1, features.shape[1], device=device), torch.zeros(1, features.shape[1], device=device)
                    if feats_1.shape[0] > 1:
                        mean_1, variance_1 = feats_1.mean(dim=0, keepdim=True), feats_1.var(dim=0, unbiased=False, keepdim=True)
                    else:
                        mean_1, variance_1 = torch.zeros(1, features.shape[1], device=device), torch.zeros(1, features.shape[1], device=device)
                    relative_variance_0 = variance_0 / (mean_0 ** 2 + 1e-6)
                    relative_variance_1 = variance_1 / (mean_1 ** 2 + 1e-6)
                    std_sum = (relative_variance_0.sqrt() + relative_variance_1.sqrt()).squeeze()
                    Relative_std += std_sum.mean()
                    max_Relative_std += std_sum.max()
                Relative_std = Relative_std / 80.0
                max_Relative_std = max_Relative_std / 80.0
                loss_cls = criterion(logits, labels)
                loss = loss_cls - vweights_avg * Relative_std - vweights_max * max_Relative_std
            elif 'MultiCrossEntropyLoss' in str(type(criterion)):
                # logits, loss = criterion(model, perturb_img, labels, optimizer)
                logits, features = model(perturb_img, return_representation=True)
                y = [(label + model.task_outputs[i]//2) % model.task_outputs[i] for i, label in enumerate(labels)]
                self.feats = dict()
                for ii in range(logits.shape[1]):
                    self.feats[ii] = dict()
                    self.feats[ii][0] = torch.tensor([]).to(device)
                    self.feats[ii][1] = torch.tensor([]).to(device)
                for j in range(len(images)):
                    for ii in range(logits.shape[1]):
                        if labels[j, ii] == 0:
                            self.feats[ii][0] = torch.cat([self.feats[ii][0], features[j].unsqueeze(0)], dim=0)
                        if labels[j, ii] == 1:
                            self.feats[ii][1] = torch.cat([self.feats[ii][1], features[j].unsqueeze(0)], dim=0)
                Relative_std = 0
                max_Relative_std = 0
                for ii in range(logits.shape[1]):
                    variance_0 = self.feats[ii][0].var(dim=0, unbiased=False).squeeze()
                    variance_1 = self.feats[ii][1].var(dim=0, unbiased=False).squeeze()
                    mean_0 = self.feats[ii][0].mean(dim=0, keepdim=True)
                    mean_1 = self.feats[ii][1].mean(dim=0, keepdim=True)
                    variance_0 = self.feats[ii][0].var(dim=0, keepdim=True)
                    variance_1 = self.feats[ii][1].var(dim=0, keepdim=True)
                    relative_variance_0 = variance_0 / (mean_0 ** 2 + 1e-6)
                    relative_variance_1 = variance_1 / (mean_1 ** 2 + 1e-6)
                    Relative_std = Relative_std + (relative_variance_0 ** 0.5 + relative_variance_1 ** 0.5).squeeze().mean()
                    max_Relative_std = max_Relative_std + (relative_variance_0 ** 0.5 + relative_variance_1 ** 0.5).squeeze().max()

                loss_cls = criterion(logits, y)
                loss = - loss_cls - vweights_avg * Relative_std - vweights_max * max_Relative_std
            loss.backward()

            eta = self.step_size * perturb_img.grad.data.sign()
            perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
            eta = torch.clamp(perturb_img.data - images.data, -self.epsilon, self.epsilon)
            perturb_img = Variable(images.data + eta, requires_grad=True)
            perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)

        print (loss_cls.item())
        print (Relative_std.item())
        print (max_Relative_std.item())
        return perturb_img, eta

    def sep_binary(self, images, labels, model, criterion, checkpoint_path_file, Features_mean, n_tasks, random_noise=None, sample_wise=False, vweights_avg = 1.0, vweights_max = 1.0):
        if random_noise is None:
            random_noise = torch.FloatTensor(*images.shape).uniform_(-self.epsilon, self.epsilon).to(device)
        perturb_img = Variable(images.data + random_noise, requires_grad=True)
        perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)
        eta = random_noise
        
        if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
            y = (labels + 1) % 2 # B * N
        elif 'MultiCrossEntropyLoss' in str(type(criterion)):
            y = [(label + model.task_outputs[i]//2) % model.task_outputs[i] for i, label in enumerate(labels)]
        for _ in range(self.num_steps):
            grads = torch.tensor([]).to(device)
            for ckpt_epoch in range(0, 60, 4):
                checkpoint = util.load_model(filename=checkpoint_path_file.replace('.pth', '_epoch{}.pth'.format(ckpt_epoch)),
                                         model=model,
                                         optimizer=None,
                                         alpha_optimizer=None,
                                         scheduler=None)
                model.eval()
                opt = torch.optim.SGD([perturb_img], lr=1e-3)
                opt.zero_grad()
                model.zero_grad()
                logits, features = model(perturb_img, return_representation=True)
                from collections import defaultdict
                self.feats = defaultdict(lambda: {0: [], 1: []})
                for j in range(len(images)):
                    for ii in range(logits.shape[1]):
                        self.feats[ii][int(labels[j, ii])].append(features[j])
                Relative_std = 0
                max_Relative_std = 0
                for ii in range(logits.shape[1]):
                    feats_0 = torch.stack(self.feats[ii][0]) if self.feats[ii][0] else torch.empty(0, features.shape[1], device=device)
                    feats_1 = torch.stack(self.feats[ii][1]) if self.feats[ii][1] else torch.empty(0, features.shape[1], device=device)
                    if feats_0.shape[0] > 1:
                        mean_0, variance_0 = feats_0.mean(dim=0, keepdim=True), feats_0.var(dim=0, unbiased=False, keepdim=True)
                    else:
                        mean_0, variance_0 = torch.zeros(1, features.shape[1], device=device), torch.zeros(1, features.shape[1], device=device)
                    if feats_1.shape[0] > 1:
                        mean_1, variance_1 = feats_1.mean(dim=0, keepdim=True), feats_1.var(dim=0, unbiased=False, keepdim=True)
                    else:
                        mean_1, variance_1 = torch.zeros(1, features.shape[1], device=device), torch.zeros(1, features.shape[1], device=device)
                    relative_variance_0 = variance_0 / (mean_0 ** 2 + 1e-6)
                    relative_variance_1 = variance_1 / (mean_1 ** 2 + 1e-6)
                    std_sum = (relative_variance_0.sqrt() + relative_variance_1.sqrt()).squeeze()
                    Relative_std += std_sum.mean()
                    max_Relative_std += std_sum.max()
                Relative_std = Relative_std / 80.0
                max_Relative_std = max_Relative_std / 80.0
                loss_cls = criterion(logits, y)
                loss = loss_cls + vweights_avg * Relative_std + vweights_max * max_Relative_std
                loss.backward()
                grads = torch.cat([grads, perturb_img.grad.data.clone().detach().unsqueeze(0)], dim=0)
            grad_avg = grads.mean(0)
            grad_sum = grad_avg
            eta = self.step_size * grad_sum.sign()
            perturb_img = Variable(perturb_img.data - eta, requires_grad=True)
            eta = torch.clamp(perturb_img.data - images.data, -self.epsilon, self.epsilon)
            perturb_img = Variable(images.data + eta, requires_grad=True)
            perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)
        print (loss_cls.item())
        print (Relative_std.item())
        print (max_Relative_std.item())
        return perturb_img, eta

    def _patch_noise_extend_to_img(self, noise, image_size=[3, 32, 32], patch_location='center'):
        c, h, w = image_size[0], image_size[1], image_size[2]
        mask = np.zeros((c, h, w), np.float32)
        x_len, y_len = noise.shape[1], noise.shape[1]

        if patch_location == 'center' or (h == w == x_len == y_len):
            x = h // 2
            y = w // 2
        elif patch_location == 'random':
            x = np.random.randint(x_len // 2, w - x_len // 2)
            y = np.random.randint(y_len // 2, h - y_len // 2)
        else:
            raise('Invalid patch location')

        x1 = np.clip(x - x_len // 2, 0, h)
        x2 = np.clip(x + x_len // 2, 0, h)
        y1 = np.clip(y - y_len // 2, 0, w)
        y2 = np.clip(y + y_len // 2, 0, w)
        if type(noise) is np.ndarray:
            pass
        else:
            mask[:, x1: x2, y1: y2] = noise.cpu().numpy()
        return ((x1, x2, y1, y2), torch.from_numpy(mask).to(device))


