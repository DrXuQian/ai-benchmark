import numpy as np
import os

# 最简单的配置
batch = 1
num_levels = 4
channels = 32
spatial_shapes = [(92, 160), (46, 80), (23, 40), (12, 20)]

level_start_index = [0]
for i, (h, w) in enumerate(spatial_shapes[:-1]):
    level_start_index.append(level_start_index[-1] + (h+2) * (w+2))
level_start_index = np.array(level_start_index, dtype=np.int64)

spatial_size = level_start_index[-1] + (spatial_shapes[-1][0]+2) * (spatial_shapes[-1][1]+2)

# 生成value数据：编码为 level*10000 + h*100 + w + c*0.01
value = np.zeros((batch, spatial_size, channels), dtype=np.float16)

for b in range(batch):
    offset = 0
    for l, (h, w) in enumerate(spatial_shapes):
        h_padded, w_padded = h + 2, w + 2
        for hi in range(h_padded):
            for wi in range(w_padded):
                for c in range(channels):
                    val = l * 10000 + hi * 100 + wi + c * 0.01
                    value[b, offset + hi * w_padded + wi, c] = val
        offset += h_padded * w_padded

value_flat = value.reshape(-1)

os.makedirs('working_simple', exist_ok=True)
value_flat.astype(np.float16).tofile('working_simple/value.bin')
level_start_index.tofile('working_simple/level_start_index.bin')
np.array(spatial_shapes, dtype=np.int64).reshape(-1).tofile('working_simple/spatial_shapes.bin')

# 只有1个query，32个points，每个point都在固定位置(h=10, w=20)
num_query = 1  
num_points = 32

sampling_loc = np.zeros((batch, num_query, num_levels, num_points, 2), dtype=np.float16)
sampling_loc[:, :, :, :, 0] = 10.5  # h
sampling_loc[:, :, :, :, 1] = 20.5  # w
sampling_loc.tofile('working_simple/sampling_locations.bin')

attn_weight = np.ones((batch, num_query, num_levels, num_points), dtype=np.float16)
attn_weight.tofile('working_simple/attention_weights.bin')

print(f"简化测试数据生成完成")
print(f"num_query=1, 所有采样点都在 (h=10.5, w=20.5)")
print(f"预期值: level=0时，hLow=10, wLow=20")
print(f"  c=0应该是: 0*10000 + 10*100 + 20 + 0*0.01 = 1020.00")
print(f"  c=1应该是: 0*10000 + 10*100 + 20 + 1*0.01 = 1020.01")
print(f"  c=7应该是: 0*10000 + 10*100 + 20 + 7*0.01 = 1020.07")

