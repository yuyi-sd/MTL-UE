import torch
import torch.nn as nn
import torch.nn.functional as F
import models
from torch.autograd import Variable
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class MadrysLoss(nn.Module):
    def __init__(self, step_size=0.007, epsilon=0.031, perturb_steps=10, distance='l_inf', cutmix=False):
        super(MadrysLoss, self).__init__()
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.distance = distance
        self.cross_entropy = models.CutMixCrossEntropyLoss() if cutmix else torch.nn.CrossEntropyLoss()

    def forward(self, model, x_natural, y, optimizer):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # generate adversarial example
        x_adv = x_natural.clone() + self.step_size * torch.randn(x_natural.shape).to(device)
        if self.distance == 'l_inf':
            for _ in range(self.perturb_steps):
                x_adv.requires_grad_()
                loss_ce = self.cross_entropy(model(x_adv), y)
                grad = torch.autograd.grad(loss_ce, [x_adv])[0]
                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        else:
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        for param in model.parameters():
            param.requires_grad = True

        model.train()
        # x_adv = Variable(x_adv, requires_grad=False)
        optimizer.zero_grad()
        logits = model(x_adv)
        loss = self.cross_entropy(logits, y)

        return logits, loss

class MultiCrossEntropyLoss:
    def __init__(self, weight=None):
        self.criterion = nn.CrossEntropyLoss()
        self.weight = weight
    
    def __call__(self, logits_list, labels_list):
        total_loss = 0
        num_groups = len(logits_list)
        if self.weight is None:
            self.weight = [1.0] * num_groups

        for logits, labels, weight in zip(logits_list, labels_list, self.weight):
            loss = self.criterion(logits, labels)
            total_loss += loss * weight
        
        weighted_average_loss = total_loss / sum(self.weight)
        return weighted_average_loss

class MultiMadrysLoss(nn.Module):
    def __init__(self, step_size=0.007, epsilon=0.031, perturb_steps=10, distance='l_inf'):
        super(MultiMadrysLoss, self).__init__()
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.distance = distance
        self.cross_entropy = MultiCrossEntropyLoss()

    def forward(self, model, x_natural, y, optimizer):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # generate adversarial example
        x_adv = x_natural.clone() + self.step_size * torch.randn(x_natural.shape).to(device)
        if self.distance == 'l_inf':
            for _ in range(self.perturb_steps):
                x_adv.requires_grad_()
                loss_ce = self.cross_entropy(model(x_adv), y)
                grad = torch.autograd.grad(loss_ce, [x_adv])[0]
                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        else:
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        for param in model.parameters():
            param.requires_grad = True

        model.train()
        # x_adv = Variable(x_adv, requires_grad=False)
        optimizer.zero_grad()
        logits = model(x_adv)
        loss = self.cross_entropy(logits, y)

        return logits, loss

class BCEMadrysLoss(nn.Module):
    def __init__(self, step_size=0.007, epsilon=0.031, perturb_steps=10, distance='l_inf'):
        super(BCEMadrysLoss, self).__init__()
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.distance = distance
        self.cross_entropy = torch.nn.BCEWithLogitsLoss()

    def forward(self, model, x_natural, y, optimizer):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # generate adversarial example
        x_adv = x_natural.clone() + self.step_size * torch.randn(x_natural.shape).to(device)
        if self.distance == 'l_inf':
            for _ in range(self.perturb_steps):
                x_adv.requires_grad_()
                loss_ce = self.cross_entropy(model(x_adv), y)
                grad = torch.autograd.grad(loss_ce, [x_adv])[0]
                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        else:
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        for param in model.parameters():
            param.requires_grad = True

        model.train()
        # x_adv = Variable(x_adv, requires_grad=False)
        optimizer.zero_grad()
        logits = model(x_adv)
        loss = self.cross_entropy(logits, y)

        return logits, loss

import math

class MultiCrossEntropySuppressLoss(nn.Module):
    def __init__(self, lambda_reg=1):
        super(MultiCrossEntropySuppressLoss, self).__init__()
        self.cross_entropy = MultiCrossEntropyLoss()
        self.lambda_reg = lambda_reg

    def entropy_suppress_loss(self, shared_feats, task_logits_list):
        """
        shared_feats: (B, d)
        task_logits_list: list of (B, Ck) logits per task
        """
        B, d = shared_feats.shape
        shared_feats = d * torch.nn.functional.normalize(shared_feats, p=1, dim=1)
        K = len(task_logits_list)
        # Compute entropy per task per sample
        entropies = torch.stack([
            -torch.sum(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1), dim=-1)
            for logits in task_logits_list
        ], dim=1)  # shape: (B, K)
        # Normalize entropy to [0, 1] by dividing by log(C_k)
        normalized_entropy = []
        for i, logits in enumerate(task_logits_list):
            C = logits.shape[1]
            norm = entropies[:, i] / math.log(C)
            normalized_entropy.append(norm)
        normalized_entropy = torch.stack(normalized_entropy, dim=1)  # (B, K)
        # Compute feature energy per sample
        feat_energy = shared_feats.pow(2).unsqueeze(1)  # (B, 1, d)
        # Compute loss: (1 - entropy) * energy
        weight = 1.0 - normalized_entropy.unsqueeze(2)  # (B, K, 1)
        loss = (weight * feat_energy).mean()
        return self.lambda_reg * loss

    def forward(self, model, x_natural, y, optimizer):
        optimizer.zero_grad()
        logits, shared_feats = model(x_natural, return_representation=True)
        task_loss = self.cross_entropy(logits, y)
        # Entropy-aware regularization
        entropy_loss = self.entropy_suppress_loss(shared_feats, logits)
        loss = task_loss + entropy_loss
        return logits, loss

import random
class MultiMadrysLoss_modified(nn.Module):
    def __init__(self, step_size=0.007, epsilon=0.031, perturb_steps=10, distance='l_inf'):
        super(MultiMadrysLoss_modified, self).__init__()
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.distance = distance
        self.cross_entropy = MultiCrossEntropyLoss()

    def forward(self, model, x_natural, y, optimizer):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        k = 1
        x_adv = x_natural.clone() + self.step_size * torch.randn(x_natural.shape).to(device)
        if self.distance == 'l_inf':
            for _ in range(self.perturb_steps):
                x_adv.requires_grad_()
                # zipped = list(zip(model(x_adv), y))
                # selected_pairs = random.sample(zipped, k)
                # selected_list1, selected_list2 = zip(*selected_pairs)
                selected_list1 = model(x_adv)[0]
                selected_list2 = y[0]
                loss_ce = self.cross_entropy(selected_list1, selected_list2)
                grad = torch.autograd.grad(loss_ce, [x_adv])[0]
                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        else:
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        for param in model.parameters():
            param.requires_grad = True

        model.train()
        optimizer.zero_grad()
        logits = model(x_adv)
        loss = self.cross_entropy(logits, y)

        return logits, loss


class MultiMadrysLossUnsupervisedPGD(nn.Module):
    def __init__(self, step_size=0.007, epsilon=0.031, perturb_steps=10, distance='l_inf', feature_distance='l2'):
        super(MultiMadrysLossUnsupervisedPGD, self).__init__()
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.distance = distance
        self.feature_distance = feature_distance
        self.cross_entropy = MultiCrossEntropyLoss()

    def forward(self, model, x_natural, y, optimizer):
        device = x_natural.device

        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # get natural features (no gradient needed)
        with torch.no_grad():
            _, feats_nat = model(x_natural, return_representation=True)

        # initialize adversarial example
        x_adv = x_natural.clone() + self.step_size * torch.randn_like(x_natural).to(device)

        if self.distance == 'l_inf':
            for _ in range(self.perturb_steps):
                x_adv.requires_grad_()
                _, feats_adv = model(x_adv, return_representation=True)

                # unsupervised objective: maximize feature difference
                if self.feature_distance == 'l2':
                    feat_diff = feats_adv - feats_nat.detach()
                    loss_feat = (feat_diff ** 2).sum()
                elif self.feature_distance == 'cosine':
                    cos_sim = nn.functional.cosine_similarity(feats_adv, feats_nat.detach(), dim=-1)
                    loss_feat = -cos_sim.mean()  
                elif self.feature_distance == 'ortho':
                    cos_sim = nn.functional.cosine_similarity(feats_adv, feats_nat.detach(), dim=-1)
                    loss_feat = -cos_sim.abs().mean()  

                grad = torch.autograd.grad(loss_feat, [x_adv])[0]
                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        else:
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        for param in model.parameters():
            param.requires_grad = True

        model.train()
        logits_adv = model(x_adv)
        loss = self.cross_entropy(logits_adv, y)

        optimizer.zero_grad()
        return logits_adv, loss


class MultiMadrysLossUnsupervisedFGSM(nn.Module):
    def __init__(self, epsilon=0.031, distance='l_inf', feature_distance='l2'):
        super(MultiMadrysLossUnsupervisedFGSM, self).__init__()
        self.step_size = epsilon * 1.2
        self.epsilon = epsilon
        self.distance = distance
        self.feature_distance = feature_distance
        self.cross_entropy = MultiCrossEntropyLoss()

    def forward(self, model, x_natural, y, optimizer):
        device = x_natural.device

        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # get natural features (no gradient needed)
        with torch.no_grad():
            _, feats_nat = model(x_natural, return_representation=True)

        # initialize adversarial example
        x_adv = x_natural.clone()

        if self.distance == 'l_inf':
            x_adv.requires_grad_()
            _, feats_adv = model(x_adv, return_representation=True)

            # unsupervised objective: maximize feature difference
            if self.feature_distance == 'l2':
                feat_diff = feats_adv - feats_nat.detach()
                loss_feat = (feat_diff ** 2).sum()
            elif self.feature_distance == 'cosine':
                cos_sim = nn.functional.cosine_similarity(feats_adv, feats_nat.detach(), dim=-1)
                loss_feat = -cos_sim.mean()  # negative cosine similarity → maximize angular difference

            grad = torch.autograd.grad(loss_feat, [x_adv])[0]
            x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        else:
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        for param in model.parameters():
            param.requires_grad = True

        model.train()
        logits_adv = model(x_adv)
        loss = self.cross_entropy(logits_adv, y)

        optimizer.zero_grad()
        return logits_adv, loss


class MultiMadrysLossFeatRegularizePGD(nn.Module):
    def __init__(self, step_size=0.007, epsilon=0.031, perturb_steps=10, distance='l_inf'):
        super(MultiMadrysLossFeatRegularizePGD, self).__init__()
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.distance = distance
        self.cross_entropy = MultiCrossEntropyLoss()

    def forward(self, model, x_natural, y, optimizer):
        device = x_natural.device

        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # get natural features (no gradient needed)
        with torch.no_grad():
            _, feats_nat = model(x_natural, return_representation=True)

        # initialize adversarial example
        x_adv = x_natural.clone() + self.step_size * torch.randn_like(x_natural).to(device)

        if self.distance == 'l_inf':
            for _ in range(self.perturb_steps):
                x_adv.requires_grad_()
                _, feats_adv = model(x_adv, return_representation=True)

                # unsupervised objective: maximize feature difference
                cos_sim = nn.functional.cosine_similarity(feats_adv, feats_nat.detach(), dim=-1)
                loss_feat = -cos_sim.mean()  # negative cosine similarity → maximize angular difference

                grad = torch.autograd.grad(loss_feat, [x_adv])[0]
                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        else:
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        for param in model.parameters():
            param.requires_grad = True

        model.train()
        logits = model(x_natural)
        loss_cls = self.cross_entropy(logits, y)
        _, feats_adv = model(x_adv, return_representation=True)
        loss_reg = 1 - nn.functional.cosine_similarity(feats_adv, feats_nat.detach(), dim=-1).mean()
        lambda_reg = 1.0
        loss = loss_cls + lambda_reg * loss_reg
        optimizer.zero_grad()
        return logits, loss



class MultiMadrysLossUnsupervisedDynamicPGD(nn.Module):
    def __init__(self, step_size=0.007, epsilon=0.031, perturb_steps=10, distance='l_inf', feature_distance='l2'):
        super(MultiMadrysLossUnsupervisedDynamicPGD, self).__init__()
        self.step_size = step_size
        self.epsilon_max = epsilon  # save as max epsilon
        self.perturb_steps = perturb_steps
        self.distance = distance
        self.feature_distance = feature_distance
        self.cross_entropy = MultiCrossEntropyLoss()

    def compute_gradient_conflict(self, model, x, y_list):
        """
        Compute gradient conflict between tasks for ResNetMTL model.
        model: ResNetMTL instance
        x: input batch
        y_list: list of target tensors, one per task
        """
        encoder_params = list(model.shared_parameters())  # get shared encoder parameters
        grads = []
        for k, (head, target) in enumerate(zip(model.task_specific, y_list)):
            x.requires_grad = True
            logits_list, features = model(x, return_representation=True)  # get logits & shared feature
            logits_k = logits_list[k]  # logits for task k
            loss_k = F.cross_entropy(logits_k, target)
            # compute grad of loss_k w.r.t shared encoder params
            grad_k = torch.autograd.grad(loss_k, encoder_params, retain_graph=True, create_graph=False)
            grad_flat = torch.cat([g.view(-1) for g in grad_k if g is not None])
            grads.append(grad_flat.detach())
        K = len(grads)
        conflict_sum = 0.0
        count = 0
        for i in range(K):
            for j in range(i + 1, K):
                cos_sim = F.cosine_similarity(grads[i], grads[j], dim=0)
                conflict_sum += cos_sim
                count += 1

        avg_cos_sim = conflict_sum / count if count > 0 else 1.0
        gradient_conflict = 1.0 - avg_cos_sim.item()
        # return gradient_conflict/2  # scalar
        return gradient_conflict

    def forward(self, model, x_natural, y, optimizer):
        device = x_natural.device
        model.eval()
        # compute dynamic epsilon
        gradient_conflict = self.compute_gradient_conflict(model, x_natural, y)
        # dynamic_epsilon = self.epsilon_max * (1 - gradient_conflict)
        dynamic_epsilon = self.epsilon_max * gradient_conflict
        dynamic_epsilon = max(0.0, min(dynamic_epsilon, self.epsilon_max))
        for param in model.parameters():
            param.requires_grad = False
        # print dynamic epsilon for debugging
        print(f"Dynamic epsilon: {255 * dynamic_epsilon:.3f}")
        with torch.no_grad():
            _, feats_nat = model(x_natural, return_representation=True)
        # initialize adversarial example
        x_adv = x_natural.clone() + self.step_size * torch.randn_like(x_natural).to(device)
        if self.distance == 'l_inf':
            for _ in range(self.perturb_steps):
                x_adv.requires_grad_()
                _, feats_adv = model(x_adv, return_representation=True)
                # unsupervised objective: maximize feature difference
                if self.feature_distance == 'l2':
                    feat_diff = feats_adv - feats_nat.detach()
                    loss_feat = (feat_diff ** 2).sum()
                elif self.feature_distance == 'cosine':
                    cos_sim = F.cosine_similarity(feats_adv, feats_nat.detach(), dim=-1)
                    loss_feat = -cos_sim.mean()
                grad = torch.autograd.grad(loss_feat, [x_adv])[0]
                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - dynamic_epsilon), x_natural + dynamic_epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        else:
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        for param in model.parameters():
            param.requires_grad = True

        model.train()
        logits_adv = model(x_adv)
        loss = self.cross_entropy(logits_adv, y)

        optimizer.zero_grad()
        return logits_adv, loss



class MultiMadrysLossUnsupervisedDescentPGD(nn.Module):
    def __init__(self, step_size=0.007, epsilon=0.031, perturb_steps=10, distance='l_inf', feature_distance='l2'):
        super(MultiMadrysLossUnsupervisedDescentPGD, self).__init__()
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.distance = distance
        self.feature_distance = feature_distance
        self.cross_entropy = MultiCrossEntropyLoss()
        self.epoch = 0
        self.gamma = 0.5
        self.milestone = [45, 54]

    def forward(self, model, x_natural, y, optimizer):
        device = x_natural.device

        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # get natural features (no gradient needed)
        with torch.no_grad():
            _, feats_nat = model(x_natural, return_representation=True)

        # initialize adversarial example
        x_adv = x_natural.clone() + self.step_size * torch.randn_like(x_natural).to(device)

        if self.distance == 'l_inf':
            for _ in range(self.perturb_steps):
                x_adv.requires_grad_()
                _, feats_adv = model(x_adv, return_representation=True)

                # unsupervised objective: maximize feature difference
                if self.feature_distance == 'l2':
                    feat_diff = feats_adv - feats_nat.detach()
                    loss_feat = (feat_diff ** 2).sum()
                elif self.feature_distance == 'cosine':
                    cos_sim = nn.functional.cosine_similarity(feats_adv, feats_nat.detach(), dim=-1)
                    loss_feat = -cos_sim.mean()  # negative cosine similarity → maximize angular difference

                grad = torch.autograd.grad(loss_feat, [x_adv])[0]
                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        else:
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        for param in model.parameters():
            param.requires_grad = True

        model.train()
        logits_adv = model(x_adv)
        loss = self.cross_entropy(logits_adv, y)

        optimizer.zero_grad()
        return logits_adv, loss


class MultiCrossEntropyLossWithCosinePenalty:
    def __init__(self, weight=None):
        self.criterion = nn.CrossEntropyLoss()
        self.weight = weight
        self.require_feature = True
    
    def __call__(self, logits_list, labels_list, features=None, lambda_penalty=0.1):
        total_loss = 0
        num_tasks = len(logits_list)

        if self.weight is None:
            self.weight = [1.0] * num_tasks

        # Cross-entropy loss per task
        for logits, labels, weight in zip(logits_list, labels_list, self.weight):
            loss = self.criterion(logits, labels)
            total_loss += loss * weight
        
        weighted_average_loss = total_loss / sum(self.weight)

        # Multi-task cosine penalty
        if features is not None and lambda_penalty > 0:
            penalty_total = 0.0
            for task_idx, labels_for_features in enumerate(labels_list):  # loop over each task's label
                penalty_task = 0.0
                unique_classes = labels_for_features.unique()
                for c in unique_classes:
                    idx = (labels_for_features == c).nonzero(as_tuple=True)[0]
                    if idx.numel() < 2:
                        continue  # skip if class has less than 2 samples
                    feats_c = features[idx]  # (Nc, D)
                    # pairwise cosine similarity matrix
                    cos_matrix = F.cosine_similarity(feats_c.unsqueeze(1), feats_c.unsqueeze(0), dim=-1)  # (Nc, Nc)
                    # mask diagonal (self-similarity)
                    mask = ~torch.eye(cos_matrix.size(0), dtype=torch.bool, device=cos_matrix.device)
                    penalty_c = cos_matrix[mask].mean()  # average off-diagonal cosine
                    penalty_task += penalty_c
                if len(unique_classes) > 0:
                    penalty_task /= len(unique_classes)  # average over classes
                penalty_total += penalty_task
            penalty_total /= num_tasks  # average over tasks

            # encourage diversity → minimize cosine similarity → subtract penalty
            weighted_average_loss = weighted_average_loss - lambda_penalty * penalty_total
        
        return weighted_average_loss





def topk_conflict_mask(feats, task_losses, topk_ratio=0.3):
    B, C = feats.shape
    grads = []
    for loss in task_losses:
        grad = torch.autograd.grad(loss, feats, retain_graph=True)[0]
        grads.append(F.normalize(grad, dim=1))
    grads = torch.stack(grads, dim=1)  # (B, T, C)
    T = grads.shape[1]
    sim_sum = 0.0
    count = 0
    for i in range(T):
        for j in range(i + 1, T):
            sim = (grads[:, i] * grads[:, j]).sum(dim=1)  # (B,)
            conflict = 1 - sim
            sim_sum += conflict.unsqueeze(1)
            count += 1
    conflict_score = sim_sum / count  # (B, 1)
    grad_std = grads.std(dim=1)       # (B, C)
    conflict = grad_std * conflict_score  # (B, C)

    k = int(C * topk_ratio)
    _, topk_idx = torch.topk(conflict, k=k, dim=1)
    mask = torch.zeros_like(conflict)
    for i in range(B):
        mask[i, topk_idx[i]] = 1.0
    return mask.detach()


def topk_activation_mask(feats, topk_ratio=0.3):
    B, C = feats.shape
    importance = feats.abs()
    k = int(C * topk_ratio)
    _, topk_idx = torch.topk(importance, k=k, dim=1)
    mask = torch.zeros_like(importance)
    for i in range(B):
        mask[i, topk_idx[i]] = 1.0
    return mask.detach()


def topk_uniform_mask(feats, topk_ratio=0.3):
    B, C = feats.shape
    k = int(C * topk_ratio)
    mask = torch.zeros_like(feats)
    for i in range(B):
        idx = torch.randperm(C)[:k]
        mask[i, idx] = 1.0
    return mask.detach()


class MultiMadrysLossUnsupervisedMaskedPGD(nn.Module):
    def __init__(self, step_size=0.007, epsilon=0.031, perturb_steps=10,
                 distance='l_inf', topk_ratio=0.01, mask_type='activation', feature_distance='l2'):
        super(MultiMadrysLossUnsupervisedMaskedPGD, self).__init__()
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.distance = distance
        self.topk_ratio = topk_ratio
        self.mask_type = mask_type
        self.cross_entropy = MultiCrossEntropyLoss()
        self.feature_distance = feature_distance

    def forward(self, model, x_natural, y, optimizer):
        device = x_natural.device

        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        if self.mask_type == 'conflict':
            x_natural.requires_grad_(True)  # VERY IMPORTANT: enable full backward trace
            logits_list, feats_nat = model(x_natural, return_representation=True)

            # IMPORTANT: make feats_nat a leaf tensor that requires grad
            feats_nat.retain_grad()  # enable autograd grad_fn trace

            losses = [F.cross_entropy(logit, label) for logit, label in zip(logits_list, y)]
            mask = topk_conflict_mask(feats_nat, losses, self.topk_ratio)
        else:
            with torch.no_grad():
                _, feats_nat = model(x_natural, return_representation=True)
            if self.mask_type == 'activation':
                mask = topk_activation_mask(feats_nat, self.topk_ratio)
            elif self.mask_type == 'random':
                mask = topk_uniform_mask(feats_nat, self.topk_ratio)


        # Generate adversarial example
        x_adv = x_natural.clone() + self.step_size * torch.randn_like(x_natural).to(device)

        for _ in range(self.perturb_steps):
            x_adv.requires_grad_()
            _, feats_adv = model(x_adv, return_representation=True)  # (B, C)
            if self.feature_distance == 'l2':
                feat_diff = (feats_adv - feats_nat.detach()) * mask  # (B, C)
                loss_feat = (feat_diff ** 2).sum() / (mask.sum() + 1e-6)
            elif self.feature_distance == 'cosine':
                # compute cosine similarity only on masked features
                f1 = feats_adv * mask
                f2 = feats_nat.detach() * mask
                cos_sim = F.cosine_similarity(f1, f2, dim=-1)  # (B,)
                loss_feat = -cos_sim.mean()
            elif self.feature_distance == 'ortho':
                # encourage orthogonality: abs cosine similarity small
                f1 = feats_adv * mask
                f2 = feats_nat.detach() * mask
                cos_sim = F.cosine_similarity(f1, f2, dim=-1)  # (B,)
                loss_feat = -cos_sim.abs().mean()
            elif self.feature_distance == 'anti_align':
                f1 = (feats_adv * mask).view(feats_adv.size(0), -1)
                f2 = (feats_nat.detach() * mask).view(feats_nat.size(0), -1)
                cos_sim = F.cosine_similarity(f1, f2, dim=-1)          # (B,)
                l2_dist = F.pairwise_distance(f1, f2, p=2)              # (B,)
                loss_feat = (-cos_sim * l2_dist).mean()
            else:
                raise ValueError(f"Unsupported feature_distance: {self.feature_distance}")

            grad = torch.autograd.grad(loss_feat, [x_adv])[0]
            if self.distance == 'l_inf':
                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
            elif self.distance == 'l2':
                epsilon = self.epsilon * (x_natural.shape[2]/32)
                step_size = self.step_size * (x_natural.shape[2]/32)
                grad_norm = torch.norm(grad.view(grad.shape[0], -1), p=2, dim=1).view(-1, 1, 1, 1)
                grad_normalized = grad / (grad_norm + 1e-10)
                x_adv = x_adv.detach() + step_size * grad_normalized
                eta = x_adv - x_natural
                eta = eta.renorm(p=2, dim=0, maxnorm=epsilon)
                x_adv = torch.clamp(x_natural + eta, 0.0, 1.0)
            else:
                raise ValueError(f"Unsupported distance: {self.distance}")

        for param in model.parameters():
            param.requires_grad = True

        model.train()
        logits = model(x_adv)
        loss = self.cross_entropy(logits, y)

        optimizer.zero_grad()
        return logits, loss



class MadrysLossUnsupervisedMaskedPGD(nn.Module):
    def __init__(self, step_size=0.007, epsilon=0.031, perturb_steps=10,
                 distance='l_inf', topk_ratio=0.01, mask_type='activation', feature_distance='l2'):
        super(MadrysLossUnsupervisedMaskedPGD, self).__init__()
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.distance = distance
        self.topk_ratio = topk_ratio
        self.mask_type = mask_type
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.feature_distance = feature_distance

    def forward(self, model, x_natural, y, optimizer):
        device = x_natural.device

        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        with torch.no_grad():
            _, feats_nat = model(x_natural, return_representation=True)
        if self.mask_type == 'activation':
            mask = topk_activation_mask(feats_nat, self.topk_ratio)
        elif self.mask_type == 'random':
            mask = topk_uniform_mask(feats_nat, self.topk_ratio)

        # Generate adversarial example
        x_adv = x_natural.clone() + self.step_size * torch.randn_like(x_natural).to(device)

        for _ in range(self.perturb_steps):
            x_adv.requires_grad_()
            _, feats_adv = model(x_adv, return_representation=True)  # (B, C)
            if self.feature_distance == 'l2':
                feat_diff = (feats_adv - feats_nat.detach()) * mask  # (B, C)
                loss_feat = (feat_diff ** 2).sum() / (mask.sum() + 1e-6)
            elif self.feature_distance == 'cosine':
                # compute cosine similarity only on masked features
                f1 = feats_adv * mask
                f2 = feats_nat.detach() * mask
                cos_sim = F.cosine_similarity(f1, f2, dim=-1)  # (B,)
                loss_feat = -cos_sim.mean()
            elif self.feature_distance == 'ortho':
                # encourage orthogonality: abs cosine similarity small
                f1 = feats_adv * mask
                f2 = feats_nat.detach() * mask
                cos_sim = F.cosine_similarity(f1, f2, dim=-1)  # (B,)
                loss_feat = -cos_sim.abs().mean()
            elif self.feature_distance == 'anti_align':
                f1 = (feats_adv * mask).view(feats_adv.size(0), -1)
                f2 = (feats_nat.detach() * mask).view(feats_nat.size(0), -1)
                cos_sim = F.cosine_similarity(f1, f2, dim=-1)          # (B,)
                l2_dist = F.pairwise_distance(f1, f2, p=2)              # (B,)
                loss_feat = (-cos_sim * l2_dist).mean()
            else:
                raise ValueError(f"Unsupported feature_distance: {self.feature_distance}")

            grad = torch.autograd.grad(loss_feat, [x_adv])[0]
            if self.distance == 'l_inf':
                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
            elif self.distance == 'l2':
                epsilon = self.epsilon * (x_natural.shape[2]/32)
                step_size = self.step_size * (x_natural.shape[2]/32)
                grad_norm = torch.norm(grad.view(grad.shape[0], -1), p=2, dim=1).view(-1, 1, 1, 1)
                grad_normalized = grad / (grad_norm + 1e-10)
                x_adv = x_adv.detach() + step_size * grad_normalized
                eta = x_adv - x_natural
                eta = eta.renorm(p=2, dim=0, maxnorm=epsilon)
                x_adv = torch.clamp(x_natural + eta, 0.0, 1.0)
            else:
                raise ValueError(f"Unsupported distance: {self.distance}")

        for param in model.parameters():
            param.requires_grad = True

        model.train()
        logits = model(x_adv)
        loss = self.cross_entropy(logits, y)

        optimizer.zero_grad()
        return logits, loss


class MadrysLossMasked(nn.Module):
    def __init__(self, step_size=0.007, epsilon=0.031, perturb_steps=10, distance='l_inf', cutmix=False, topk_ratio=0.004, mask_type='activation'):
        super(MadrysLossMasked, self).__init__()
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.distance = distance
        self.cross_entropy = models.CutMixCrossEntropyLoss() if cutmix else torch.nn.CrossEntropyLoss()
        self.topk_ratio = topk_ratio
        self.mask_type = mask_type

    def forward(self, model, x_natural, y, optimizer):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        with torch.no_grad():
            _, feats_nat = model(x_natural, return_representation=True)
        if self.mask_type == 'activation':
            mask = topk_activation_mask(feats_nat, self.topk_ratio)
        elif self.mask_type == 'random':
            mask = topk_uniform_mask(feats_nat, self.topk_ratio)

        # generate adversarial example
        x_adv = x_natural.clone() + self.step_size * torch.randn(x_natural.shape).to(device)
        for _ in range(self.perturb_steps):
            x_adv.requires_grad_()
            loss_ce = self.cross_entropy(model(x_adv, mask = mask), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            if self.distance == 'l_inf':
                    x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
                    x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
                    x_adv = torch.clamp(x_adv, 0.0, 1.0)
            elif self.distance == 'l2':
                    epsilon = self.epsilon * (x_natural.shape[2]/32)
                    step_size = self.step_size * (x_natural.shape[2]/32)
                    grad_norm = torch.norm(grad.view(grad.shape[0], -1), p=2, dim=1).view(-1, 1, 1, 1)
                    grad_normalized = grad / (grad_norm + 1e-10)
                    x_adv = x_adv.detach() + step_size * grad_normalized
                    eta = x_adv - x_natural
                    eta = eta.renorm(p=2, dim=0, maxnorm=epsilon)
                    x_adv = torch.clamp(x_natural + eta, 0.0, 1.0)
            else:
                raise ValueError(f"Unsupported distance: {self.distance}")

        for param in model.parameters():
            param.requires_grad = True

        model.train()
        # x_adv = Variable(x_adv, requires_grad=False)
        optimizer.zero_grad()
        logits = model(x_adv)
        loss = self.cross_entropy(logits, y)

        return logits, loss


