import numpy as np
import struct
import os

# 使用working目录的配置
batch = 1
num_levels = 4
channels = 32
spatial_shapes = [(92, 160), (46, 80), (23, 40), (12, 20)]

# 计算每个level的起始索引
level_start_index = [0]
for i, (h, w) in enumerate(spatial_shapes[:-1]):
    level_start_index.append(level_start_index[-1] + (h+2) * (w+2))
level_start_index = np.array(level_start_index, dtype=np.int64)

spatial_size = level_start_index[-1] + (spatial_shapes[-1][0]+2) * (spatial_shapes[-1][1]+2)

print(f"batch={batch}, spatial_size={spatial_size}, num_levels={num_levels}")
print(f"level_start_index: {level_start_index}")
print(f"spatial_shapes: {spatial_shapes}")

# 生成value数据：每个位置的值 = batch*100000 + level*10000 + h*100 + w + c*0.01
value = np.zeros((batch, spatial_size, channels), dtype=np.float16)

for b in range(batch):
    offset = 0
    for l, (h, w) in enumerate(spatial_shapes):
        h_padded, w_padded = h + 2, w + 2
        for hi in range(h_padded):
            for wi in range(w_padded):
                for c in range(channels):
                    # 编码：batch*100000 + level*10000 + h*100 + w + c*0.01
                    val = b * 100000 + l * 10000 + hi * 100 + wi + c * 0.01
                    value[b, offset + hi * w_padded + wi, c] = val
        offset += h_padded * w_padded

# 展平为1D
value_flat = value.reshape(-1)

# 保存为.bin文件
os.makedirs('working_pattern', exist_ok=True)
value_flat.astype(np.float16).tofile('working_pattern/value.bin')
level_start_index.tofile('working_pattern/level_start_index.bin')
np.array(spatial_shapes, dtype=np.int64).reshape(-1).tofile('working_pattern/spatial_shapes.bin')

# 复制sampling_loc和attention_weights（使用简单值）
num_query = 10
num_points = 32

# sampling_loc: 简单的位置，level=0, 位置在(h=10, w=20)附近
sampling_loc = np.zeros((batch, num_query, num_levels, num_points, 2), dtype=np.float16)
for b in range(batch):
    for q in range(num_query):
        for l in range(num_levels):
            for p in range(num_points):
                # 设置为固定位置方便调试
                sampling_loc[b, q, l, p, 0] = 10.5  # h
                sampling_loc[b, q, l, p, 1] = 20.5  # w

sampling_loc.tofile('working_pattern/sampling_locations.bin')

# attention_weights: 全1
attn_weight = np.ones((batch, num_query, num_levels, num_points), dtype=np.float16)
attn_weight.tofile('working_pattern/attention_weights.bin')

print(f"\n生成的测试数据:")
print(f"- value.bin: {value_flat.shape} elements")
print(f"- 每个值编码为: batch*100000 + level*10000 + h*100 + w + c*0.01")
print(f"\n示例: batch=0, level=0, h=10, w=20, c=5 的值应该是: 0*100000 + 0*10000 + 10*100 + 20 + 5*0.01 = 1020.05")
print(f"\n如果TMA加载正确，在(h=10, w=20)位置，c=0通道应该看到: 1020.00")
print(f"c=1通道应该看到: 1020.01, c=2通道应该看到: 1020.02, 以此类推")

