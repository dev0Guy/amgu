import torch
import torchattacks
from torch import nn
import matplotlib.pyplot as plt
import foolbox

class Attacks(object):

    @classmethod
    def from_str(cls,name: str)-> object:
        if name == "FGSM":
            return cls.FGSM
        elif name == "HopSkipJump":
            return cls.HopSkipJump
        elif name == "GN":
            return cls.GN
        elif name == "BIM":
            return cls.BIM
        elif name == "CW":
            return cls.CW
        elif name == "RFGSM":
            return cls.RFGSM
        elif name == "PGD":
            return cls.PGD
        elif name == "PGDL2":
            return cls.PGDL2
        elif name == "EOTPGD":
            return cls.EOTPGD
        elif name == "TPGD":
            return cls.TPGD
        elif name == "FFGSM":
            return cls.FFGSM
        elif name == "MIFGSM":
            return cls.MIFGSM
        elif name == "APGD":
            return cls.APGD
        elif name == "APGDT":
            return cls.APGDT
        elif name == "FAB":
            return cls.FAB
        elif name == "Square":
            return cls.Square
        elif name == "AutoAttack":
            return cls.AutoAttack
        elif name == "OnePixel":
            return cls.OnePixel
        elif name == "DeepFool":
            return cls.DeepFool
        elif name == "SparseFool":
            return cls.SparseFool
        elif name == "DIFGSM":
            return cls.DIFGSM
        elif name == "UPGD":
            return cls.UPGD
        elif name == "TIFGSM":
            return cls.TIFGSM
        elif name == "Jitter":
            return cls.Jitter
        elif name == "Pixle":
            return cls.Pixle
        else: 
            return cls.defualt
    
    @staticmethod
    def defualt(model: nn.Module, imgs: torch.Tensor, target: torch.Tensor):
        return imgs

    @staticmethod
    def GN(model: nn.Module, imgs: torch.Tensor,target: torch.Tensor):
        model.eval()
        attack = torchattacks.GN(model)
        return attack(imgs,model(imgs))

    @staticmethod
    def BIM(model: nn.Module, imgs: torch.Tensor,target: torch.Tensor):
        model.eval()
        attack = torchattacks.BIM(model, eps=4/255, alpha=1/255, steps=0)
        return attack(imgs,model(imgs))

    @staticmethod
    def CW(model: nn.Module, imgs: torch.Tensor,target: torch.Tensor):
        model.eval()
        attack = torchattacks.CW(model, c=1e-4, kappa=0, steps=1000, lr=0.01)
        return attack(imgs,model(imgs))

    @staticmethod
    def RFGSM(model: nn.Module, imgs: torch.Tensor,target: torch.Tensor):
        model.eval()
        attack = torchattacks.RFGSM(model, eps=16/255, alpha=8/255, steps=1)
        return attack(imgs,model(imgs))

    @staticmethod
    def PGD(model: nn.Module, imgs: torch.Tensor,target: torch.Tensor):
        model.eval()
        attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=True)
        return attack(imgs,model(imgs))

    @staticmethod
    def PGDL2(model: nn.Module, imgs: torch.Tensor,target: torch.Tensor):
        model.eval()
        attack = torchattacks.PGDL2(model, eps=1.0, alpha=0.2, steps=40, random_start=True)
        return attack(imgs,model(imgs))

    @staticmethod
    def EOTPGD(model: nn.Module, imgs: torch.Tensor,target: torch.Tensor):
        model.eval()
        attack = torchattacks.EOTPGD(model, eps=4/255, alpha=8/255, steps=40, eot_iter=10)
        return attack(imgs,model(imgs))

    @staticmethod
    def TPGD(model: nn.Module, imgs: torch.Tensor,target: torch.Tensor):
        model.eval()
        attack = torchattacks.TPGD(model, eps=8/255, alpha=2/255, steps=7)
        return attack(imgs)

    @staticmethod
    def FFGSM(model: nn.Module, imgs: torch.Tensor,target: torch.Tensor):
        model.eval()
        attack = torchattacks.FFGSM(model, eps=8/255, alpha=10/255)
        return  attack(imgs,model(imgs))

    @staticmethod
    def MIFGSM(model: nn.Module, imgs: torch.Tensor,target: torch.Tensor):
        model.eval()
        attack = torchattacks.MIFGSM(model, eps=8/255, steps=5, decay=1.0)
        return  attack(imgs,model(imgs))

    @staticmethod
    def APGD(model: nn.Module, imgs: torch.Tensor,target: torch.Tensor):
        model.eval()
        attack = torchattacks.APGD(model, norm='Linf', eps=8/255, steps=100, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False)
        return  attack(imgs,model(imgs))

    @staticmethod
    def APGDT(model: nn.Module, imgs: torch.Tensor,target: torch.Tensor):
        model.eval()
        attack = attack = torchattacks.APGDT(model, norm='Linf', eps=8/255, steps=100, n_restarts=1, seed=0, eot_iter=1, rho=.75, verbose=False, n_classes=10)
        return  attack(imgs,model(imgs))

    @staticmethod
    def FAB(model: nn.Module, imgs: torch.Tensor,target: torch.Tensor):
        model.eval()
        attack = torchattacks.FAB(model, norm='Linf', steps=100, eps=None, n_restarts=1, alpha_max=0.1, eta=1.05, beta=0.9, loss_fn=None, verbose=False, seed=0, targeted=False, n_classes=10)
        return  attack(imgs,model(imgs))

    @staticmethod
    def Square(model: nn.Module, imgs: torch.Tensor,target: torch.Tensor):
        model.eval()
        attack = torchattacks.Square(model, model, norm='Linf', n_queries=5000, n_restarts=1, eps=None, p_init=.8, seed=0, verbose=False, targeted=False, loss='margin', resc_schedule=True)
        return  attack(imgs,model(imgs))

    @staticmethod
    def AutoAttack(model: nn.Module, imgs: torch.Tensor,target: torch.Tensor):
        model.eval()
        attack = torchattacks.AutoAttack(model, norm='Linf', eps=.3, version='standard', n_classes=10, seed=None, verbose=False)
        return  attack(imgs,model(imgs))

    @staticmethod
    def OnePixel(model: nn.Module, imgs: torch.Tensor,target: torch.Tensor):
        model.eval()
        attack = torchattacks.OnePixel(model, pixels=1, steps=75, popsize=400, inf_batch=128)
        return  attack(imgs,model(imgs))

    @staticmethod
    def DeepFool(model: nn.Module, imgs: torch.Tensor,target: torch.Tensor):
        model.eval()
        attack = torchattacks.DeepFool(model, steps=50, overshoot=0.02)
        return  attack(imgs,model(imgs))

    @staticmethod
    def SparseFool(model: nn.Module, imgs: torch.Tensor,target: torch.Tensor):
        model.eval()
        attack = torchattacks.SparseFool(model, steps=20, lam=3, overshoot=0.02)
        return  attack(imgs,model(imgs))

    @staticmethod
    def DIFGSM(model: nn.Module, imgs: torch.Tensor,target: torch.Tensor):
        model.eval()
        attack =torchattacks.DI2FGSM(model, eps=8/255, alpha=2/255, steps=20, decay=0.0, resize_rate=0.9, diversity_prob=0.5, random_start=False)
        return  attack(imgs,model(imgs))

    @staticmethod
    def UPGD(model: nn.Module, imgs: torch.Tensor,target: torch.Tensor):
        model.eval()
        attack = torchattacks.UPGD(model, eps=8/255, alpha=1/255, steps=40, random_start=False)
        return  attack(imgs,model(imgs))
        
    @staticmethod
    def TIFGSM(model: nn.Module, imgs: torch.Tensor,target: torch.Tensor):
        model.eval()
        attack = torchattacks.TIFGSM(model, eps=8/255, alpha=2/255, steps=20, decay=1.0, resize_rate=0.9, diversity_prob=0.7, random_start=False)
        return  attack(imgs,model(imgs))

    @staticmethod
    def Jitter(model: nn.Module, imgs: torch.Tensor,target: torch.Tensor):
        model.eval()
        attack = torchattacks.Jitter(model, eps=0.3, alpha=2/255, steps=40, scale=10, std=0.1, random_start=True)
        return attack(imgs,model(imgs))

    @staticmethod
    def Pixle(model: nn.Module, imgs: torch.Tensor,target: torch.Tensor):
        model.eval()
        attack = torchattacks.Pixle(model, x_dimensions=(0.1, 0.2), restarts=100, iteratsion=50)
        return  attack(imgs,model(imgs))

    @staticmethod
    def FGSM(model: nn.Module, imgs: torch.Tensor,target: torch.Tensor):
        model.eval()
        attack = torchattacks.FGSM(model, eps=0.001)
        return attack(imgs,model(imgs))

    @staticmethod
    def HopSkipJump(model: nn.Module, imgs: torch.Tensor,target: torch.Tensor):
        # criterion = foolbox.criteria.TargetedMisclassification(labels)
        attack = foolbox.v1.attacks.HopSkipJumpAttack
        print(attack)
        # model.eval()
        x = attack(imgs,model(imgs))
        print(x)
        return x

