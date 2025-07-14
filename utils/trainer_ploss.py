from transformers import Trainer
import torch
import torch.nn.functional as F
class CustomTrainerWithEntropyLoss(Trainer):
    """
    自定义Trainer类，支持熵辅助损失
    
    当使用origin_model时，会计算QAT模型和原始模型输出的熵差异作为辅助损失
    """
    
    def __init__(self, origin_model=None, entropy_loss_weight=0.1, lm_loss_weight=1.0, **kwargs):
        """
        初始化自定义训练器
        
        Args:
            origin_model: 原始模型（用于计算熵损失）
            entropy_loss_weight: 熵损失的权重
            lm_loss_weight: 语言模型损失的权重
            **kwargs: 其他传递给父类的参数
        """
        super().__init__(**kwargs)
        self.origin_model = origin_model
        self.entropy_loss_weight = entropy_loss_weight
        self.lm_loss_weight = lm_loss_weight
        self.last_logged_step = -1
        # 如果有原始模型，将其设置为评估模式并冻结参数
        if self.origin_model is not None:
            self.origin_model.eval()
            for param in self.origin_model.parameters():
                param.requires_grad = False
            # 将原始模型移动到相同设备
            if hasattr(self.model, 'device'):
                self.origin_model.to(self.model.device)
    
    def compute_entropy_loss(self, qat_logits, origin_logits):
        """
        计算熵损失
        
        Args:
            qat_logits: QAT模型的logits输出
            origin_logits: 原始模型的logits输出
            
        Returns:
            entropy_loss: 熵损失值
        """
        # 计算概率分布
        # 计算概率分布
        qat_probs = F.softmax(qat_logits.float(), dim=-1)
        origin_probs = F.softmax(origin_logits.float(), dim=-1)
        
        # 计算熵
        qat_entropy = -torch.sum(qat_probs * torch.log(qat_probs + 1e-8), dim=-1)
        origin_entropy = -torch.sum(origin_probs * torch.log(origin_probs + 1e-8), dim=-1)
        
        # 计算熵差异损失（使用MSE）
        entropy_loss = F.mse_loss(qat_entropy, origin_entropy)
        
        return entropy_loss
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        重写损失计算函数，添加熵辅助损失
        
        Args:
            model: 当前训练的模型
            inputs: 输入数据
            return_outputs: 是否返回模型输出
            
        Returns:
            loss: 总损失（加权语言模型损失 + 熵损失）
            outputs: 模型输出（如果return_outputs=True）
        """
        # 获取标签
        labels = inputs.get("labels")
        
        # 前向传播得到QAT模型的输出
        outputs = model(**inputs)
        qat_logits = outputs.get("logits")
        
        # 计算原始的语言模型损失
        if labels is not None:
            # 计算交叉熵损失
            shift_logits = qat_logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        else:
            lm_loss = outputs.get("loss", torch.tensor(0.0))
        
        # 如果有原始模型且权重大于0，计算熵损失
        if self.origin_model is not None and self.entropy_loss_weight > 0:
            with torch.no_grad():
                # 原始模型前向传播
                origin_outputs = self.origin_model(**inputs)
                origin_logits = origin_outputs.get("logits")
            
            # 计算熵损失
            entropy_loss = self.compute_entropy_loss(qat_logits, origin_logits)
            
            # 总损失 = 加权语言模型损失 + 加权熵损失
            total_loss = self.lm_loss_weight * lm_loss + self.entropy_loss_weight * entropy_loss
            
            # 存储损失组件供日志记录使用（避免重复打印）
            self._current_lm_loss = lm_loss.item()
            self._current_entropy_loss = entropy_loss.item()
            self._current_total_loss = total_loss.item()
        else:
            total_loss = self.lm_loss_weight * lm_loss
            self._current_lm_loss = lm_loss.item()
            self._current_entropy_loss = 0.0
            self._current_total_loss = total_loss.item()
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def log(self, logs, start_time=None, **kwargs):
        """
        重写日志记录方法，添加自定义损失组件
        
        Args:
            logs: 日志字典
            start_time: 开始时间（可选）
            **kwargs: 其他参数
        """
        # 只在有损失组件数据时添加自定义日志
        if hasattr(self, '_current_lm_loss'):
            logs.update({
                "train_lm_loss": self._current_lm_loss,
                "train_entropy_loss": self._current_entropy_loss,
                "lm_loss_weight": self.lm_loss_weight,
                "entropy_loss_weight": self.entropy_loss_weight
            })
        
        # 调用父类的日志记录方法
        super().log(logs, start_time, **kwargs)