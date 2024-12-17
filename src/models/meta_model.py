# """
#    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/maml.py)
#    """
import torch
import torch.nn as nn

from .meta_utils import clone_module, update_module, clone_parameters
from logging import getLogger
from torch.autograd import grad

logger = getLogger()

class MetaLearner(nn.Module):

    def __init__(self, module=None):
        super().__init__()
        self.module = module

    def __getattr__(self, attr):
        try:
            return super().__getattr__(attr)
        except AttributeError:
            return getattr(self.__dict__['_modules']['module'], attr)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

class MAML(MetaLearner):
    def __init__(self,
                 module,
                 lr,
                 eta = 0,
                 first_order=False,
                 allow_unused=None,
                 clip_norm = 0,
                 allow_nograd=False):
        super().__init__(module)
        self.lr = lr
        self.eta = eta
        self.first_order = first_order
        self.allow_nograd = allow_nograd
        self.clip_norm = clip_norm
        if allow_unused is None:
            allow_unused = allow_nograd
        self.allow_unused = allow_unused

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    def clone(self, first_order=None, allow_unused=None, allow_nograd=None):
        """
        **Description**

        Returns a `MAML`-wrapped copy of the module whose parameters and buffers
        are `torch.clone`d from the original module.

        This implies that back-propagating losses on the cloned module will
        populate the buffers of the original module.
        For more information, refer to learn2learn.clone_module().

        **Arguments**

        * **first_order** (bool, *optional*, default=None) - Whether the clone uses first-
            or second-order updates. Defaults to self.first_order.
        * **allow_unused** (bool, *optional*, default=None) - Whether to allow differentiation
        of unused parameters. Defaults to self.allow_unused.
        * **allow_nograd** (bool, *optional*, default=False) - Whether to allow adaptation with
            parameters that have `requires_grad = False`. Defaults to self.allow_nograd.

        """
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        return MAML(clone_module(self.module),
                    lr=self.lr,
                    eta = self.eta,
                    first_order=first_order,
                    allow_unused=allow_unused,
                    allow_nograd=allow_nograd)



    def adapt(self,
              loss,
              lr = None,
              first_order=None,
              allow_unused=None,
              allow_nograd=None):
        """
        **Description**

        Takes a gradient step on the loss and updates the cloned parameters in place.

        **Arguments**

        * **loss** (Tensor) - Loss to minimize upon update.
        * **first_order** (bool, *optional*, default=None) - Whether to use first- or
            second-order updates. Defaults to self.first_order.
        * **allow_unused** (bool, *optional*, default=None) - Whether to allow differentiation
            of unused parameters. Defaults to self.allow_unused.
        * **allow_nograd** (bool, *optional*, default=None) - Whether to allow adaptation with
            parameters that have `requires_grad = False`. Defaults to self.allow_nograd.

        """
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        if lr is None:
            lr = self.lr
        second_order = not first_order

        if allow_nograd:
            # Compute relevant gradients
            diff_params = [p for p in self.module.parameters() if p.requires_grad]
            grad_params = grad(loss,
                               diff_params,
                               retain_graph=second_order,
                               create_graph=second_order,
                               allow_unused=allow_unused)
            gradients = []
            grad_counter = 0

            # Handles gradients for non-differentiable parameters
            for param in self.module.parameters():
                if param.requires_grad:
                    gradient = grad_params[grad_counter]
                    grad_counter += 1
                else:
                    gradient = None
                gradients.append(gradient)
        else:
            try:
                gradients = grad(loss,
                                 self.module.parameters(),
                                 retain_graph=second_order,
                                 create_graph=second_order,
                                 allow_unused=allow_unused)
            except RuntimeError:
                logger.info('learn2learn: Maybe try with allow_nograd=True and/or allow_unused=True ?')

        # Update the module
        self.module = self.maml_update(grads=gradients,lr = lr)



    def maml_update(self, grads=None,lr = None):
        """
        [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/maml.py)

        **Description**

        Performs a MAML update on model using grads and lr.
        The function re-routes the Python object, thus avoiding in-place
        operations.

        NOTE: The model itself is updated in-place (no deepcopy), but the
              parameters' tensors are not.

        **Arguments**

        * **model** (Module) - The model to update.
        * **lr** (float) - The learning rate used to update the model.
        * **grads** (list, *optional*, default=None) - A list of gradients for each parameter
            of the model. If None, will use the gradients in .grad attributes.

        **Example**
        ~~~python
        maml = l2l.algorithms.MAML(Model(), lr=0.1)
        model = maml.clone() # The next two lines essentially implement model.adapt(loss)
        grads = autograd.grad(loss, model.parameters(), create_graph=True)
        maml_update(model, lr=0.1, grads)
        ~~~
        """
        if lr is None:
            lr = self.lr
        if grads is not None:
            params = list(self.module.parameters())
            if not len(grads) == len(list(params)):
                msg = 'WARNING:maml_update(): Parameters and gradients have different length. ('
                msg += str(len(params)) + ' vs ' + str(len(grads)) + ')'
                print(msg)
            for p, g in zip(params, grads):
                if self.clip_norm> 0:
                    g = g.clamp(min=-self.clip_norm,
                                      max=self.clip_norm)
                if g is not None:
                    if self.eta ==0:
                        p.update = - lr * g
                    else:
                        p.update = - lr * g +  torch.sqrt(torch.tensor(2.0 * lr * self.eta)) * torch.randn_like(g)
        return update_module(self.module)


class MAMLAdamW(MetaLearner):
    def __init__(self,
                 module,
                 lr,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0.01,
                 first_order=False,
                 allow_unused=None,
                 allow_nograd=False):
        super().__init__(module)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.first_order = first_order
        self.allow_nograd = allow_nograd
        if allow_unused is None:
            allow_unused = allow_nograd
        self.allow_unused = allow_unused

        self.init_adamw_params()


    def init_adamw_params(self):
        """Initialize AdamW specific parameters."""
        self.m = [torch.zeros_like(p) for p in self.module.parameters()]
        self.v = [torch.zeros_like(p) for p in self.module.parameters()]
        self.t = 0  # timestep

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def clone(self, first_order=None, allow_unused=None, allow_nograd=None):
        """
        **Description**

        Returns a `MAML`-wrapped copy of the module whose parameters and buffers
        are `torch.clone`d from the original module.

        This implies that back-propagating losses on the cloned module will
        populate the buffers of the original module.
        For more information, refer to learn2learn.clone_module().

        **Arguments**

        * **first_order** (bool, *optional*, default=None) - Whether the clone uses first-
            or second-order updates. Defaults to self.first_order.
        * **allow_unused** (bool, *optional*, default=None) - Whether to allow differentiation
        of unused parameters. Defaults to self.allow_unused.
        * **allow_nograd** (bool, *optional*, default=False) - Whether to allow adaptation with
            parameters that have `requires_grad = False`. Defaults to self.allow_nograd.

        """
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        return MAMLAdamW(clone_module(self.module),
                    lr=self.lr,
                     betas=self.betas,
                     eps=self.eps,
                     weight_decay=self.weight_decay,
                    first_order=first_order,
                    allow_unused=allow_unused,
                    allow_nograd=allow_nograd)



    def adapt(self,
              loss,
              lr = None,
              first_order=None,
              allow_unused=None,
              allow_nograd=None):
        """
        **Description**

        Takes a gradient step on the loss and updates the cloned parameters in place.

        **Arguments**

        * **loss** (Tensor) - Loss to minimize upon update.
        * **first_order** (bool, *optional*, default=None) - Whether to use first- or
            second-order updates. Defaults to self.first_order.
        * **allow_unused** (bool, *optional*, default=None) - Whether to allow differentiation
            of unused parameters. Defaults to self.allow_unused.
        * **allow_nograd** (bool, *optional*, default=None) - Whether to allow adaptation with
            parameters that have `requires_grad = False`. Defaults to self.allow_nograd.

        """
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        if lr is None:
            lr = self.lr
        second_order = not first_order

        if allow_nograd:
            # Compute relevant gradients
            diff_params = [p for p in self.module.parameters() if p.requires_grad]
            grad_params = grad(loss,
                               diff_params,
                               retain_graph=second_order,
                               create_graph=second_order,
                               allow_unused=allow_unused)
            gradients = []
            grad_counter = 0

            # Handles gradients for non-differentiable parameters
            for param in self.module.parameters():
                if param.requires_grad:
                    gradient = grad_params[grad_counter]
                    grad_counter += 1
                else:
                    gradient = None
                gradients.append(gradient)
        else:
            try:
                gradients = grad(loss,
                                 self.module.parameters(),
                                 retain_graph=second_order,
                                 create_graph=second_order,
                                 allow_unused=allow_unused)
            except RuntimeError:
                logger.info('learn2learn: Maybe try with allow_nograd=True and/or allow_unused=True ?')

        # Update the module
        self.module = self.maml_update(grads=gradients,lr = lr)



    def maml_update(self, grads=None,lr = None):
        """
        [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/maml.py)

        **Description**

        Performs a MAML update on model using grads and lr.
        The function re-routes the Python object, thus avoiding in-place
        operations.

        NOTE: The model itself is updated in-place (no deepcopy), but the
              parameters' tensors are not.

        **Arguments**

        * **model** (Module) - The model to update.
        * **lr** (float) - The learning rate used to update the model.
        * **grads** (list, *optional*, default=None) - A list of gradients for each parameter
            of the model. If None, will use the gradients in .grad attributes.

        **Example**
        ~~~python
        maml = l2l.algorithms.MAML(Model(), lr=0.1)
        model = maml.clone() # The next two lines essentially implement model.adapt(loss)
        grads = autograd.grad(loss, model.parameters(), create_graph=True)
        maml_update(model, lr=0.1, grads)
        ~~~
        """
        self.t += 1
        if lr is None:
            lr = self.lr
        if grads is not None:
            params = list(self.module.parameters())
            if not len(grads) == len(list(params)):
                msg = 'WARNING:maml_update(): Parameters and gradients have different length. ('
                msg += str(len(params)) + ' vs ' + str(len(grads)) + ')'
                print(msg)
            for i, (param, grad) in enumerate(zip(self.module.parameters(), grads)):
                if grad is None:
                    continue
                self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * grad
                self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * grad * grad

                m_hat = self.m[i] / (1 - self.betas[0] ** self.t)  # Correct bias for first moment
                v_hat = self.v[i] / (1 - self.betas[1] ** self.t)  # Correct bias for second moment

                # AdamW update rule
                param.update = -lr * m_hat / (
                            torch.sqrt(v_hat) + self.eps) + self.weight_decay * param
            #
            # for p, g in zip(params, grads):
            #     if g is not None:
            #         if self.eta ==0:
            #             p.update = - lr * g
            #         else:
            #             p.update = - lr * g +  torch.sqrt(torch.tensor(2.0 * lr * self.eta)) * torch.randn_like(g)
        return update_module(self.module)


class MetaSGD(MetaLearner):
    """

    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/meta_sgd.py)

    **Description**

    High-level implementation of *Meta-SGD*.

    This class wraps an arbitrary nn.Module and augments it with `clone()` and `adapt`
    methods.
    It behaves similarly to `MAML`, but in addition a set of per-parameters learning rates
    are learned for fast-adaptation.

    **Arguments**

    * **model** (Module) - Module to be wrapped.
    * **lr** (float) - Initialization value of the per-parameter fast adaptation learning rates.
    * **first_order** (bool, *optional*, default=False) - Whether to use the first-order version.
    * **lrs** (list of Parameters, *optional*, default=None) - If not None, overrides `lr`, and uses the list
        as learning rates for fast-adaptation.

    **References**

    1. Li et al. 2017. “Meta-SGD: Learning to Learn Quickly for Few-Shot Learning.” arXiv.

    **Example**

    ~~~python
    linear = l2l.algorithms.MetaSGD(nn.Linear(20, 10), lr=0.01)
    clone = linear.clone()
    error = loss(clone(X), y)
    clone.adapt(error)
    error = loss(clone(X), y)
    error.backward()
    ~~~
    """

    def __init__(self, model, lr=0.01, first_order=False, lrs=None):
        super(MetaSGD, self).__init__()
        self.module = model
        if lrs is None:
            lrs = [torch.ones_like(p) * lr for p in model.parameters()]
            lrs = nn.ParameterList([nn.Parameter(lr) for lr in lrs])
        self.lrs = lrs
        self.first_order = first_order

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def clone(self):
        """
        **Descritpion**

        Akin to `MAML.clone()` but for MetaSGD: it includes a set of learnable fast-adaptation
        learning rates.
        """
        return MetaSGD(clone_module(self.module),
                       lrs=clone_parameters(self.lrs),
                       first_order=self.first_order)


    def adapt(self, loss, first_order=None, lr=None):
        """
        **Descritpion**

        Akin to `MAML.adapt()` but for MetaSGD: it updates the model with the learnable
        per-parameter learning rates.
        """
        if first_order is None:
            first_order = self.first_order
        second_order = not first_order
        # gradients = grad(loss,
        #                  self.module.parameters(),
        #                  retain_graph=second_order,
        #                  create_graph=second_order)

        diff_params = [p for p in self.module.parameters() if p.requires_grad]
        grad_params = grad(loss,
                           diff_params,
                           retain_graph=second_order,
                           create_graph=second_order)
        gradients = []
        grad_counter = 0

        # Handles gradients for non-differentiable parameters
        for param in self.module.parameters():
            if param.requires_grad:
                gradient = grad_params[grad_counter]
                grad_counter += 1
            else:
                gradient = None
            gradients.append(gradient)
        self.module = self.meta_sgd_update(self.module, self.lrs, gradients)

    def meta_sgd_update(self,model, lrs=None, grads=None):
        """

        **Description**

        Performs a MetaSGD update on model using grads and lrs.
        The function re-routes the Python object, thus avoiding in-place
        operations.

        NOTE: The model itself is updated in-place (no deepcopy), but the
              parameters' tensors are not.

        **Arguments**

        * **model** (Module) - The model to update.
        * **lrs** (list) - The meta-learned learning rates used to update the model.
        * **grads** (list, *optional*, default=None) - A list of gradients for each parameter
            of the model. If None, will use the gradients in .grad attributes.

        **Example**
        ~~~python
        meta = l2l.algorithms.MetaSGD(Model(), lr=1.0)
        lrs = [th.ones_like(p) for p in meta.model.parameters()]
        model = meta.clone() # The next two lines essentially implement model.adapt(loss)
        grads = autograd.grad(loss, model.parameters(), create_graph=True)
        meta_sgd_update(model, lrs=lrs, grads)
        ~~~
        """
        if grads is not None and lrs is not None:
            for p, lr, g in zip(model.parameters(), lrs, grads):
                p.grad = g
                p._lr = lr

        # Update the params
        for param_key in model._parameters:
            p = model._parameters[param_key]
            if p is not None and p.grad is not None:
                model._parameters[param_key] = p - p._lr * p.grad
                p.grad = None
                p._lr = None

        # Second, handle the buffers if necessary
        for buffer_key in model._buffers:
            buff = model._buffers[buffer_key]
            if buff is not None and buff.grad is not None and buff._lr is not None:
                model._buffers[buffer_key] = buff - buff._lr * buff.grad
                buff.grad = None
                buff._lr = None

        # Then, recurse for each submodule
        for module_key in model._modules:
            model._modules[module_key] = self.meta_sgd_update(model._modules[module_key])
        return model


# if __name__ == '__main__':
#     import numpy as np
#     from torch import optim
#
#     simple_model = nn.Linear(20,10)
#     meta_model = MAML(simple_model, lr = 0.1)
#
#     meta_lr = 0.09
#     X = torch.from_numpy(np.random.randn(100,20)).float()
#     y = torch.from_numpy(np.random.randn(100,10)).float()
#     opt = optim.Adam(meta_model.parameters(), meta_lr)
#
#     num_iteration = 10
#     num_task = 10
#     num_updates = 5
#
#     for i in range(num_iteration):
#         error = 0
#         for j in range(num_task):
#             learner = meta_model.clone()
#             X_this = X[i * 10:(i + 1) * 10, :]
#             y_this = y[i * 10:(i + 1) * 10, :]
#             for _ in range(num_updates):
#                 error = torch.sum((learner(X_this[:8,:])-y_this[:8,:])**2)
#                 learner.adapt(error)
#             valid_error = (learner(X_this[8:,:])-y_this[8:,:]).detach().numpy()
#             valid_error = np.linalg.norm(valid_error)/ np.linalg.norm((y_this[8:,:]).detach().numpy())
#             valid_error/=2
#             error += valid_error
#         error/=num_task
#
#         opt.zero_grad()
#         error.backward()
#         opt.step()
#     print(error)


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import numpy as np
    import copy


    def print_model_parameters(model, msg="Model parameters"):
        print(msg)
        for param in model.parameters():
            print(param.data)

    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(1, 40),
                nn.ReLU(),
                nn.Linear(40, 1)
            )

        def forward(self, x):
            return self.net(x)

    class Train:
        def __init__(self,model):
            self.model = model

        def sample_sine_task(self,amplitude_range, phase_range, num_samples=10):
            amplitude = np.random.uniform(*amplitude_range)
            phase = np.random.uniform(*phase_range)
            inputs = np.random.uniform(-5, 5, size=(num_samples, 1))
            outputs = amplitude * np.sin(inputs + phase)
            return torch.tensor(inputs, dtype=torch.float32), torch.tensor(outputs, dtype=torch.float32)


        def maml_train(self, epochs, num_tasks, num_inner_updates, inner_lr, outer_lr):
            model = self.model
            optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)
            loss_fn = nn.MSELoss()

            for epoch in range(epochs):
                meta_loss = 0.0

                print_model_parameters(model, "Original model parameters before inner updates")
                for _ in range(num_tasks):
                    # Sample a new task
                    # learner = model.clone()
                    inputs, targets = self.sample_sine_task(amplitude_range=(0.1, 5.0),
                                                       phase_range=(0, np.pi))

                    # Clone model for inner updates
                    learner = copy.deepcopy(model)
                    print_model_parameters(learner, "Learner parameters before inner updates")
                    # Inner loop: Task-specific adaptation
                    inner_optimizer = torch.optim.SGD(learner.parameters(), lr=inner_lr)
                    for _ in range(num_inner_updates):
                        predictions = learner(inputs)
                        loss = loss_fn(predictions, targets)
                        # learner.adapt(loss)
                        learner.zero_grad()
                        loss.backward()
                        inner_optimizer.step()
                    print_model_parameters(learner, "Learner parameters after inner updates")
                    print_model_parameters(model, "Original model parameters after inner updates")
                    # Outer loop: Meta update
                    meta_predictions = learner(inputs)
                    task_meta_loss = loss_fn(meta_predictions, targets)
                    meta_loss += task_meta_loss
                optimizer.zero_grad()
                meta_loss /= num_tasks
                meta_loss.backward()
                for model_param, learner_param in zip(model.parameters(), learner.parameters()):
                    if model_param.grad is None:
                        model_param.grad = learner_param.grad.clone()
                    else:
                        model_param.grad += learner_param.grad.clone()
                optimizer.step()
                print_model_parameters(model, "Original model parameters after outer updates")

                # Average the accumulated meta-loss and update the meta model


            print(f"Epoch {epoch + 1}: Meta Loss: {meta_loss.item()}")

    # Instantiate model and start training
    model = SimpleModel()
    train = Train(model)
    train.maml_train(epochs=1, num_tasks=2, num_inner_updates=1, inner_lr=0.01, outer_lr=0.1)
