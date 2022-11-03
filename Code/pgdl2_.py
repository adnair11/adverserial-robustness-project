import torch
import torch.nn as nn

from torchattacks.attack import Attack
import rep_transformations as rt


class PGDL2_(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]
    Distance Measure : L2
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 1.0)
        alpha (float): step size. (Default: 0.2)
        steps (int): number of steps. (Default: 40)
        random_start (bool): using random initialization of delta. (Default: True)
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    Examples::
        >>> attack = torchattacks.PGDL2(model, eps=1.0, alpha=0.2, steps=40, random_start=True)
        >>> adv_images = attack(images, labels)
    """
    def __init__(self, model, transf=None, eps=1.0, alpha=0.2, steps=20,
                 random_start=True, eps_for_division=1e-10):
        if transf is None:
            self.transf = rt.Identity()
        else:
            self.transf = transf
        super().__init__("PGDL2_"+self.transf.name, model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.eps_for_division = eps_for_division
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
        batch_size = len(images)

        if self.random_start:
            # Starting at a uniformly random point
            delta = torch.empty_like(adv_images_transf).normal_()
            d_flat = delta.view(adv_images_transf.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(adv_images_transf.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*self.eps
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
            if grad.dtype is torch.cfloat:
                grad_norm = torch.sqrt((grad.real)**2 + (grad.imag)**2) + self.eps_for_division
                grad = grad/grad_norm
            else:
                grad_norm = torch.norm(grad, p='fro', dim=[-2,-1], keepdim=True) + self.eps_for_division
                grad = grad/grad_norm
            adv_images_transf = adv_images_transf.detach() + self.alpha*grad

            delta = adv_images_transf - images_transf
            if delta.dtype is torch.cfloat:
                delta_norm = torch.sqrt((delta.real)**2 + (delta.imag)**2) + self.eps_for_division
                factor = self.eps/delta_norm
                factor = torch.min(factor, torch.ones_like(delta_norm))
                delta *= factor
            else:
                delta_norm = torch.norm(delta, p='fro', dim=[-2,-1], keepdim=True)
                factor = self.eps/delta_norm
                factor = torch.min(factor, torch.ones_like(delta_norm))
                delta *= factor

            adv_images_transf = (images_transf + delta).detach()
            adv_images = torch.clamp(self.transf.inv(adv_images_transf), min=0, max=1)
            adv_images_transf = self.transf(adv_images).detach()

        return adv_images.detach()