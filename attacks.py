import torch
import torchattacks
from torch import nn

__all__ = ['Attacks']


class Attacks(object):


    @staticmethod
    def whitebox_untargeted(model: nn.Module, imgs: torch.Tensor,target: torch.Tensor,_preprocess):
        model.eval()
        imgs = _preprocess(imgs)
        attack = torchattacks.GN(model,std=0.000000000000001)
        return attack(imgs,model(imgs))


    @staticmethod
    def whitebox_targeted(model: nn.Module, imgs: torch.Tensor,target: torch.Tensor,_preprocess):
        model.eval()
        imgs = _preprocess(imgs)
        attack = torchattacks.FAB(model, norm='Linf', steps=1000, eps=0.0, n_restarts=1, alpha_max=0.1, eta=1.05, beta=0.9, verbose=False, seed=0, targeted=True, n_classes=8)
        return  attack.attack_single_run_targeted(imgs, target)

    @staticmethod
    def blackbox_targeted(model: nn.Module, imgs: torch.Tensor,target: torch.Tensor,_preprocess):
        model.eval()
        imgs = _preprocess(imgs)
        attack = torchattacks.Square(model, norm='Linf', n_queries=5000, n_restarts=1, eps=0.01, p_init=.3, seed=0, verbose=False, loss='margin', resc_schedule=True)
        return  attack(imgs,target)


    @staticmethod
    def blackbox_untargeted(model: nn.Module, imgs: torch.Tensor,target: torch.Tensor,_preprocess):
        model.eval()
        imgs = _preprocess(imgs)
        attack = torchattacks.Pixle(model)
        return  attack(imgs,imgs)

    # @staticmethod
    # def FGSM(model: nn.Module, imgs: torch.Tensor,target: torch.Tensor):
    #     model.eval()
    #     attack = torchattacks.FGSM(model, eps=0.001)
    #     return attack(imgs,model(imgs))
