import torch
import torch.nn as nn

from torchattacks.attack import Attack
import rep_transformations as rt


class FGSM_(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]
    Distance Measure : Linf
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.007)
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    Examples::
        >>> attack = torchattacks.FGSM(model, eps=0.007)
        >>> adv_images = attack(images, labels)
    """
    def __init__(self, model, transf=None, eps=0.007):
        if transf is None:
            self.transf = rt.Identity()
        else:
            self.transf = transf
        super().__init__("FGSM_"+self.transf.name, model)
        self.eps = eps
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
        
        adv_images_transf = images_transf.clone().detach().to(self.device)
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

        adv_images_transf = images_transf + self.eps*grad.sgn()
        adv_images = torch.clamp(self.transf.inv(adv_images_transf), min=0, max=1).detach()

        return adv_images