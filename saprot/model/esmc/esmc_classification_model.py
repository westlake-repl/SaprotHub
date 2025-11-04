# ==================== 临时诊断代码 ====================
# 请用这段代码替换您原来的 ESMCClassificationModel 类定义

import torchmetrics
import torch
import sys  # 导入 sys 模块以安全退出
from torch.nn.functional import cross_entropy
from torch.nn.utils.rnn import pad_sequence

# 假设这些导入路径在您的项目中是正确的
# 您需要确保下面的导入路径与您的项目结构相符
from saprot.model.model_interface import register_model
from saprot.model.esmc.base import ESMCBaseModel

try:
    from esm.sdk.api import ESMProtein
except ImportError:
    raise ImportError("Please install esm package first: pip install esm")


@register_model
class ESMCClassificationModel(ESMCBaseModel):
    def __init__(self, num_labels: int, **kwargs):
        """
        这是一个临时的、用于诊断的 __init__ 方法。
        """
        print("--- 进入诊断版 __init__ ---")
        self.num_labels = num_labels
        
        # 调用父类，这会创建 self.model 对象
        print("正在调用 super().__init__() 来加载模型...")
        super().__init__(task="classification", **kwargs)
        print("模型加载完成！")
        
        print("\n--- 开始打印 self.model 的诊断信息 ---")
        
        # 1. 打印 self.model 的类型
        print(f"1. self.model 的类型是: {type(self.model)}")
        
        # 2. 打印 self.model 的所有属性和方法 (这是最重要的！)
        #    我们将在这里寻找 'config', 'params', 'embed_dim' 等关键词
        print("\n2. 使用 dir(self.model) 打印所有可用属性和方法:")
        print(dir(self.model))
        
        # 3. 根据 dir() 的输出，我们猜测配置信息可能在 'model_config_' 中，尝试打印它
        if hasattr(self.model, 'model_config_'):
            print("\n3. 发现 'model_config_' 属性，正在打印其内容...")
            config_dict = self.model.model_config_
            print(f"   - 'model_config_' 的类型是: {type(config_dict)}")
            print(f"   - 'model_config_' 包含的键 (keys): {list(config_dict.keys())}")
            if 'embed_dim' in config_dict:
                print(f"   - ✅ 成功找到 'embed_dim' 键！其值为: {config_dict['embed_dim']}")
            else:
                print("   - ❌ 在 'model_config_' 字典中未找到 'embed_dim' 键。")
        else:
            print("\n3. 未找到 'model_config_' 属性。")

        print("\n--- 诊断完成 ---")
        
        # 安全地停止程序，防止它因为后续的错误而崩溃
        sys.exit("诊断已完成，程序已安全停止。请检查上面的输出。")

    # --- 以下的方法暂时不会被执行，可以保持原样 ---
    def initialize_metrics(self, stage):
        return {f"{stage}_acc": torchmetrics.Accuracy(task="multiclass", num_classes=self.num_labels)}

    def forward(self, inputs, coords=None):
        pass
    
    def loss_func(self, stage, logits, labels):
        pass

    def on_validation_epoch_end(self):
        pass

    def get_log_dict(self, stage):
        pass

    def reset_metrics(self, stage):
        pass

# ==================== 诊断代码结束 ====================
