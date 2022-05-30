import torch
import torchattacks
from torch import nn

class Attacks(object):

    @staticmethod
    def run_on_intersections(input,attack,labels):
        cpy = torch.zeros(input.size())
        for intersection_id in range(input.size()[1]):
            cpy[:,intersection_id] = attack(input[:,intersection_id],labels[intersection_id])
        return cpy * 255


    @staticmethod
    def whitebox_untargeted(model: nn.Module, imgs: torch.Tensor,target: torch.Tensor,preprocess):
        model.eval()
        imgs = preprocess.transform(imgs)
        attack = torchattacks.GN(model,std=0.01)
        labels = model(imgs[None,:])
        return Attacks.run_on_intersections(imgs,attack,labels)


    @staticmethod
    def whitebox_targeted(model: nn.Module, imgs: torch.Tensor,target: torch.Tensor,preprocess):
        model.eval()
        imgs = preprocess.transform(imgs)
        attack = torchattacks.FAB(model, norm='Linf', steps=1000, eps=0.0, n_restarts=1, alpha_max=0.1, eta=1.05, beta=0.9, verbose=False, seed=0, targeted=True, n_classes=8)
        return Attacks.run_on_intersections(imgs,attack,target)

    @staticmethod
    def blackbox_targeted(model: nn.Module, imgs: torch.Tensor,target: torch.Tensor,preprocess):
        model.eval()
        imgs = preprocess.transform(imgs)
        attack = torchattacks.Square(model, norm='Linf', n_queries=5000, n_restarts=1, eps=0.01, p_init=.3, seed=0, verbose=False, loss='margin', resc_schedule=True)
        return Attacks.run_on_intersections(imgs,attack,target)

    @staticmethod
    def blackbox_untargeted(model: nn.Module, imgs: torch.Tensor,target: torch.Tensor,preprocess):
        model.eval()
        imgs = preprocess.transform(imgs)
        attack = torchattacks.Pixle(model)
        labels = model(imgs[None,:])
        return Attacks.run_on_intersections(imgs,attack,labels)
