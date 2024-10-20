import matplotlib.pyplot as plt
import numpy as np

# 從 .npy 檔案讀取數據
commanded_positions = np.load('/home/peter/isaacgym/python/examples/commanded_positions.npy')
observed_positions = np.load('/home/peter/isaacgym/python/examples/observed_positions.npy')
time_steps = np.load('/home/peter/isaacgym/python/examples/time_steps.npy')
# 假設時間序列
# time_steps = np.linspace(0, 10, len(commanded_positions))  # 這裡假設總時間為 10 秒

# 繪製控制命令與觀察到的位置
plt.figure(figsize=(10, 6))
plt.plot(time_steps, commanded_positions, label='Commanded Position')
plt.plot(time_steps, observed_positions, label='Observed Position')
plt.xlabel('Time (s)')
plt.ylabel('Position (rad)')
plt.title('Commanded vs Observed Positions')
plt.legend()
plt.grid(True)
plt.show()
