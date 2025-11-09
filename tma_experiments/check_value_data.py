import numpy as np

value = np.fromfile('working_simple/value.bin', dtype=np.float16)
print(f"Total elements: {len(value)}")
print(f"First 64 values (first 2 spatial positions, 32 channels each):")
for i in range(64):
    if i % 32 == 0:
        print(f"\nPosition {i//32} (should be h=0, w={i//32}, level=0):")
    if i % 8 == 0:
        print(f"  c={i%32}-{min((i%32)+7, 31)}: ", end="")
    print(f"{value[i]:.2f} ", end="")
    if (i % 8 == 7):
        print()

print("\n\n检查 h=0, w=0, c=0-7: 预期全是 0.00, 0.01, 0.02, ..., 0.07")
print(f"实际: {value[0:8]}")

print("\n检查 h=0, w=1, c=0-7: 预期全是 1.00, 1.01, 1.02, ..., 1.07")  
print(f"实际: {value[32:40]}")

