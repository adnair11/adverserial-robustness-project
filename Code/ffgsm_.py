import torch
import torch.nn as nn

from torchattacks.attack import Attack
import rep_transformations as rt


class FFGSM_(Attack):
    r"""
    New FGSM proposed in 'Fast is better than free: Revisiting adversarial training'
    [https://arxiv.org/abs/2001.03994]
    Distance Measure : Linf
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 10/255)
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`, `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    Examples::
        >>> attack = torchattacks.FFGSM(model, eps=8/255, alpha=10/255)
        >>> adv_images = attack(images, labels)
    """
    def __init__(self, model, transf=None, eps=8/255, alpha=10/255):
        if transf is None:
            self.transf = rt.Identity()
        else:
            self.transf = transf
        super().__init__("FFGSM_"+self.transf.name, model)
        self.eps = eps
        self.alpha = alpha
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

        adv_images_transf = (images_transf + torch.randn_like(images_transf).uniform_(-self.eps, self.eps)).detach()
        adv_images_transf.requires_grad = True
        adv_images = torch.clamp(self.transf.inv(adv_images_transf), min=0, max=1)

        outputs = self.model(adv_images)

        # Calculate loss
        if self.targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(cost, adv_images_transf,
                                   retain_graph=False, create_graph=False)[0]

        adv_images_transf = adv_images_transf + self.alpha*grad.sgn()
        
        delta = adv_images_transf - images_transf
        if delta.dtype is torch.cfloat:
            delta_real = torch.clamp(delta.real, min=-self.eps, max=self.eps)
            delta_imag = torch.clamp(delta.imag, min=-self.eps, max=self.eps)
            delta = torch.complex(delta_real, delta_imag)
        else:
            delta = torch.clamp(delta, min=-self.eps, max=self.eps)
            
        adv_images = torch.clamp(self.transf.inv(images_transf + delta), min=0, max=1).detach()

        return adv_images