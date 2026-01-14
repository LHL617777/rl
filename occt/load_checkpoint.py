import torch
import numpy as np
import os
import sys


# ===================== 依赖检查 =====================
try:
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    print("错误：请先安装 omegaconf (运行 pip install omegaconf)")
    sys.exit(1)

# ===================== 配置区域 =====================
# 填入你要统计的 5 个 Checkpoint 路径
CHECKPOINT_PATHS = [
    "E:\\rl\\occt\\outputs\\2026-01-11\\16-18-33\\checkpoints_occt\\checkpoint_4096000_frames.pt",
    "E:\\rl\\occt\\outputs\\2026-01-11\\16-18-33\\checkpoints_occt\\checkpoint_4087808_frames.pt",
    "E:\\rl\\occt\\outputs\\2026-01-11\\16-18-33\\checkpoints_occt\\checkpoint_4075520_frames.pt",
    "E:\\rl\\occt\\outputs\\2026-01-11\\16-18-33\\checkpoints_occt\\checkpoint_4067328_frames.pt",
    "E:\\rl\\occt\\outputs\\2026-01-11\\16-18-33\\checkpoints_occt\\checkpoint_4055040_frames.pt"
]

# ===================== 核心逻辑 =====================
def main():
    means = []
    vars = []
    
    print(f"开始读取 {len(CHECKPOINT_PATHS)} 个 Checkpoint...\n")

    for path in CHECKPOINT_PATHS:
        if not os.path.exists(path):
            print(f"⚠️ 文件不存在，跳过: {path}")
            continue
            
        try:
            # weights_only=False 允许加载包含配置对象的完整字典
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            
            m = ckpt.get("vecnorm_mean")
            v = ckpt.get("vecnorm_var")
            
            if m is not None and v is not None:
                means.append(m)
                vars.append(v)
                print(f"✅ 已加载: {os.path.basename(path)}")
            else:
                print(f"❌ 数据缺失: {os.path.basename(path)} (无 VecNorm 数据)")
                
        except Exception as e:
            print(f"❌ 读取错误 {path}: {e}")

    if not means:
        print("\n❌ 未能提取到任何有效的 VecNorm 数据。")
        return

    # === 计算平均值 ===
    # stack 堆叠成 (N, 12) 然后沿 axis=0 求平均
    avg_mean = np.mean(np.array(means), axis=0)
    avg_var = np.mean(np.array(vars), axis=0)

    print("\n" + "="*80)
    print("📊 统计结果 (已平均)")
    print("="*80)
    
    # 打印可直接复制的代码块
    print("\n请将以下代码复制并替换到 TwoCarrierEnv.__init__ 中：\n")
    
    # 使用 repr() 确保浮点数精度完整保留，tolist() 转为标准列表格式
    print(f"# === 固化的观测归一化参数 (来自 {len(means)} 个模型的平均值) ===")
    print(f"self.vecnorm_mean = np.array({avg_mean.tolist()}, dtype=np.float64)")
    print(f"self.vecnorm_var = np.array({avg_var.tolist()}, dtype=np.float64)")
    print("self.vecnorm_frozen = True  # 强制使用固定统计量")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()