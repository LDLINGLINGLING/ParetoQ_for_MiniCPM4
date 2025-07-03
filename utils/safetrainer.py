from transformers import default_data_collator, Trainer
import torch
from utils import utils
log = utils.get_logger("clm")
class SafeTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nan_count = 0
        self.max_nan_tolerance = 5  # 最大容忍NaN次数
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        重写compute_loss方法，添加NaN检测和处理
        """
        # 调用父类的compute_loss方法
        if return_outputs:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        else:
            loss = super().compute_loss(model, inputs, return_outputs=False)
            outputs = None
        
        # 检查loss是否为NaN
        if torch.isnan(loss) or torch.isinf(loss):
            self.nan_count += 1
            log.warning(f"NaN/Inf loss detected! Count: {self.nan_count}/{self.max_nan_tolerance}")
            
            if self.nan_count >= self.max_nan_tolerance:
                raise ValueError(f"Training stopped due to {self.nan_count} consecutive NaN losses")
            
            # 返回一个小的正值替代NaN
            loss = torch.tensor(0.01, device=loss.device, dtype=loss.dtype, requires_grad=True)
        else:
            # 重置NaN计数器
            self.nan_count = 0
        
        # 添加数值稳定性检查
        if loss.item() < 1e-8:
            log.warning(f"Loss is extremely small: {loss.item()}, adding small epsilon")
            loss = loss + 1e-8
        
        return (loss, outputs) if return_outputs else loss
    
    
    
    