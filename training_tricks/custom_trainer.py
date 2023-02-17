import math
from typing import Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim.lr_scheduler import _LRScheduler
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback


#------------------------------------------------------------
# Focal Loss

class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

#------------------------------------------------------------
# Custom implementation of Cosine Annealing Warm Up Restarts

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)


    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2 for base_lr in self.base_lrs]


    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr



#------------------------------------------------------------
# Custom HuggingFace Trainer

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, device=torch.device('cuda')):
        # forward pass
        labels = inputs.pop("labels").to(torch.int64)
        
        logit = model(**inputs)

        # focal loss
        criterion = {
            'loss' : FocalLoss().to(device)
        }

        loss = criterion['loss'](logit, labels)
        outputs = torch.argmax(logit, dim = 1)

        return (loss, outputs) if return_outputs else loss



#------------------------------------------------------------
# Main

import gc

def main(
    model, 
    train_dataset, 
    val_dataset, 
    compute_metrics, 
    stop:int=3, 
    batch:int=16, 
    lr:float=1e-5, 
    epoch:int=5, 
    fold_index:int=0
):
    args = TrainingArguments(run_name = f'fold_{fold_index}',
                             output_dir= f"fold_{fold_index}",                      # 모델저장경로
                             evaluation_strategy="steps",                           # 모델의 평가를 언제 진행할지
                             eval_steps=10,                                         # 500 스텝 마다 모델 평가
                             logging_steps=10,
                             per_device_train_batch_size=batch,                        # GPU에 학습데이터를 몇개씩 올려서 학습할지
                             per_device_eval_batch_size=batch,                         # GPU에 학습데이터를 몇개씩 올려서 평가할지
                             gradient_accumulation_steps=16,
                             num_train_epochs=epoch,                                  # 전체 학습 진행 횟수
                             learning_rate=lr,                                        # 학습률 정의 
                             seed=42,                                                 
                             load_best_model_at_end=True,                           # 평가기준 스코어가 좋은 모델만 저장할지 여부
                             fp16=True,
                             do_train=True,
                             do_eval=True,
                             save_steps=10,
                             save_total_limit = 2,                                  # 저장할 모델의 갯수
                             # metric_for_best_model
                             # greater_is_better = True,
    )
    trainer = CustomTrainer(model=model,
                            args=args,
                            train_dataset=train_dataset,                      # 학습데이터
                            eval_dataset=val_dataset,                        # validation 데이터
                            compute_metrics=compute_metrics,                       # 모델 평가 방식
                            callbacks=[EarlyStoppingCallback(early_stopping_patience=stop)],)

    trainer.train()

    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
