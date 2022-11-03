import torch
import torch.nn as nn

from torchattacks.attack import Attack
import rep_transformations as rt


class PGD_(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]
    Distance Measure : Linf
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.3)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 40)
        random_start (bool): using random initialization of delta. (Default: True)
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`, `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=True)
        >>> adv_images = attack(images, labels)
    """
    def __init__(self, model, transf=None, eps=0.3,
                 alpha=2/255, steps=20, random_start=True):
        if transf is None:
            self.transf = rt.Identity()
        else:
            self.transf = transf
        super().__init__("PGD_"+self.transf.name, model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images_transf = self.transf(images).clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self._get_target_label(images_transf, labels)

        loss = nn.CrossEntropyLoss()

        adv_images_transf = images_transf.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            delta = torch.empty_like(adv_images_transf).uniform_(-self.eps, self.eps)
            adv_images_transf = (adv_images_transf + delta)
            adv_images = torch.clamp(self.transf.inv(adv_images_transf), min=0, max=1)
            adv_images_transf = self.transf(adv_images).detach()

        for _ in range(self.steps):
            adv_images_transf.requires_grad = True
            adv_images = self.transf.inv(adv_images_transf)
            outputs = self.model(adv_images)
        
            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)
                


            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images_transf,
                                       retain_graph=False, create_graph=False)[0]

            adv_images_transf = adv_images_transf.detach() + self.alpha*grad.sgn()
            
            #Clip epsilon away from original image
            delta = adv_images_transf - images_transf
            if delta.dtype is torch.cfloat:
                delta_real = torch.clamp(delta.real, min=-self.eps, max=self.eps)
                delta_imag = torch.clamp(delta.imag, min=-self.eps, max=self.eps)
                delta = torch.complex(delta_real, delta_imag)
            else:
                delta = torch.clamp(delta, min=-self.eps, max=self.eps)
            adv_images_transf = images_transf + delta
            adv_images = torch.clamp(self.transf.inv(adv_images_transf), min=0, max=1)
            adv_images_transf = self.transf(adv_images).detach()

        return adv_images.detach()