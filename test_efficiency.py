# 分别加载transformer和mamba，然后构造随机输入，对比时间开销
import time
import torch
from transformer.Models import UnevenTransformer
from vision_mamba.Models import UnevenMamba

def choose_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _tensor_sum(x, device):
    if torch.is_tensor(x):
        return x.sum()
    if isinstance(x, (list, tuple)):
        s = None
        for el in x:
            val = _tensor_sum(el, device)
            s = val if s is None else s + val
        return s if s is not None else torch.tensor(0., device=device)
    if isinstance(x, dict):
        s = None
        for v in x.values():
            val = _tensor_sum(v, device)
            s = val if s is None else s + val
        return s if s is not None else torch.tensor(0., device=device)
    # fallback for non-tensor leaves
    return torch.tensor(0., device=device)

def benchmark_model(model, inp, runs=50, warmup=5, backward=False):
    device = inp.device
    model.to(device)
    model.train() if backward else model.eval()

    # warmup
    for _ in range(warmup):
        out = model(inp)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(inp)
        if backward:
            # 使用输出中所有 tensor 的 sum 作为简单标量 loss
            loss = _tensor_sum(out, device)
            model.zero_grad()
            loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times = sorted(times)
    # 抛弃极端值，取中间平均
    trim = int(len(times)*0.1)
    sel = times[trim:len(times)-trim] if len(times) > 2*trim else times
    return sum(sel)/len(sel), min(times), max(times)

def main():
    device = choose_device()
    print("device:", device)

    transformer_args = dict(
        n_layers=6,
        n_heads=8,
        d_k=192,
        d_v=96,
        d_model=512,
        d_inner=1024,
        pad_idx=None,
        n_position=15*15,
        dropout=0.1,
        train_shape=[12, 12],
        output_dim=10,
    )

    mamba_args = dict(
        n_layers=3,
        d_state=16,
        dt_rank=32,
        d_model=512,
        pad_idx=None,
        n_position=15*15,
        dropout=0.1,
        drop_path=0.25,
        train_shape=[12, 12],
        output_dim=10,
    )

    # 输入：batch=1, channels=6, H=100, W=100
    inp = torch.randn(1, 6, 100, 100, device=device, dtype=torch.float32)

    # 按模型顺序分别加载测试，测试间释放显存
    for name, ModelClass, args in [
        ("Transformer", UnevenTransformer, transformer_args),
        ("Mamba", UnevenMamba, mamba_args),
    ]:
        print(f"\n{name}: 实例化并 benchmark ...")
        model = ModelClass(**args)  # 先在 CPU 上构造，benchmark_model 会移动到 device
        print("Warmup & benchmark forward only...")
        t_fwd, t_min, t_max = benchmark_model(model, inp, runs=200, warmup=5, backward=False)
        print(f"{name} forward avg: {t_fwd:.6f}s (min {t_min:.6f}, max {t_max:.6f})")

        print("Benchmark forward + backward (single-step backward)...")
        t_fb, _, _ = benchmark_model(model, inp, runs=200, warmup=3, backward=True)
        print(f"{name} fwd+bwd avg: {t_fb:.6f}s")

        # 释放模型与显存
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()