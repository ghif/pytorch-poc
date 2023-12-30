import timeit
import torch
import numpy as np
import matplotlib.pyplot as plt

cpu_times = []
mps_times = []
m_list = [100, 300, 500, 750, 1000, 1500, 2000, 2500, 3000]
for m in m_list:
    a_cpu = torch.rand(m, device="cpu")
    B_cpu = torch.rand((m, m), device="cpu")
    a_mps = torch.rand(m, device="mps")
    B_mps = torch.rand((m, m), device="mps")

    print(f"\n m = {m}")
    cpu_time = timeit.timeit(lambda: B_cpu @ a_cpu, number=100)
    mps_time = timeit.timeit(lambda: B_mps @ a_mps, number=100)

    cpu_times.append(cpu_time)
    mps_times.append(mps_time)

    print(f"[vec @ mat] cpu: {cpu_time}")
    print(f"[vec @ mat] mps: {mps_time}")


plt.plot(m_list, cpu_times, label="cpu")

# set width of bar 
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8)) 
 
# Set position of bar on X axis 
br1 = np.arange(len(m_list)) 
br2 = [x + barWidth for x in br1] 
 
# Make the plot
plt.bar(br1, cpu_times, color ='r', width = barWidth, 
        edgecolor ='grey', label ='cpu') 
plt.bar(br2, mps_times, color ='b', width = barWidth, 
        edgecolor ='grey', label ='mps') 
 
# Adding Xticks 
plt.xlabel('Dimension', fontweight ='bold', fontsize = 15) 
plt.ylabel('Elapsed time', fontweight ='bold', fontsize = 15) 
plt.xticks([r + (barWidth/2) for r in range(len(m_list))], 
        m_list)
 
plt.legend()
plt.show() 