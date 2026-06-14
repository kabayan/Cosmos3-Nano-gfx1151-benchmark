import torch
import time
import sys

# --- Configuration ---
MATRIX_SIZE = 16384
device = torch.device('cuda:0')

def print_header():
    if not torch.cuda.is_available():
        print('Error: No GPU detected.')
        sys.exit(1)

    props = torch.cuda.get_device_properties(device)
    print(f'\n{"="*70}')
    print(f'   ROCm Radeon gfx1151 - PEAK BENCHMARK SUITE')
    print(f'{"="*70}')
    print(f'[System] GPU Model:   {props.name}')
    print(f'[System] VRAM Total:  {props.total_memory / 1024**3:.2f} GB')
    print(f'[Config] Matrix Size: {MATRIX_SIZE} x {MATRIX_SIZE}')
    print(f'{"-"*70}')

def run_test(label, dtype, use_tf32=False, iters=10):
    try:
        # allow_tf32 is not supported/needed on ROCm AMD, but we match NVIDIA test structure
        try:
            torch.backends.cuda.matmul.allow_tf32 = use_tf32
        except AttributeError:
            pass

        x = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device, dtype=dtype)
        y = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device, dtype=dtype)

        print(f'Testing: {label}...')

        # Warmup
        for _ in range(3):
            torch.mm(x, y)
        torch.cuda.synchronize()

        # Benchmark
        start_time = time.time()
        for _ in range(iters):
            torch.mm(x, y)
        torch.cuda.synchronize()
        end_time = time.time()

        total_seconds = end_time - start_time
        total_ops = 2 * (MATRIX_SIZE**3) * iters
        tflops = (total_ops / total_seconds) / 1e12
        gflops = (total_ops / total_seconds) / 1e9

        print(f'  > Time:   {total_seconds:.4f} sec (for {iters} iterations)')
        print(f'  > Result: {tflops:.4f} TFLOPS  |  {gflops:.1f} GFLOPS')
        print(f'{"-"*70}')

        del x, y
        torch.cuda.empty_cache()

    except Exception as e:
        print(f'  > Error: {e}')
        print(f'{"-"*70}')

# --- Main Execution Flow ---
print_header()

# 1. FP16
run_test(label='FP16 (Half Precision)', dtype=torch.float16, use_tf32=False, iters=50)

# 2. BF16
run_test(label='BF16 (BFloat16)', dtype=torch.bfloat16, use_tf32=False, iters=50)

# 3. FP32
run_test(label='FP32 (IEEE 754 Single)', dtype=torch.float32, use_tf32=False, iters=10)

# 4. FP64
run_test(label='FP64 (Double Precision)', dtype=torch.float64, use_tf32=False, iters=5)

print('Benchmark Complete.')
