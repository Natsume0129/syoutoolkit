import matplotlib.pyplot as plt
import numpy as np

# 设置绘图风格，确保可以显示中文（如果需要）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def draw_smiling_ranking_graph():
    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 生成时间轴数据 (x轴)
    x = np.linspace(0, 10, 100)
    
    # --- 1. 生成 True Smiling Pattern (蓝色曲线) ---
    # 使用 Sigmoid 函数模拟：起始增长慢，中间快，最后饱和
    # 公式参考: y = L / (1 + exp(-k*(x-x0)))
    y_true = 8 / (1 + np.exp(-1.5 * (x - 5)))
    
    # --- 2. 生成 Not True Smiling Pattern (黑色曲线) ---
    # 使用带微小波动的线性/对数增长模拟：起始比蓝线快，但后续增长乏力
    # 添加一点正弦波模拟手写的抖动感
    y_not_true = 1.2 * np.sqrt(x * 5) + 0.2 * np.sin(x * 3)
    
    # 调整数据让它们从原点附近出发
    y_true = y_true - y_true[0]
    y_not_true = y_not_true - y_not_true[0]

    # --- 3. 绘图 ---
    # 绘制蓝色曲线 (True Smiling Pattern)
    ax.plot(x, y_true, color='#007bff', label='True Smiling Pattern', lw=2.5)
    
    # 绘制黑色曲线 (Not True Smiling Pattern)
    ax.plot(x, y_not_true, color='black', label='Not True Smiling Pattern', lw=2)

    # --- 4. 设置坐标轴 ---
    # 隐藏上方和右方的边框，模拟手画的坐标系
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 设置坐标轴交叉点在原点
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    
    # 添加箭头
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

    # --- 5. 添加标签 ---
    # Y 轴标签
    ax.set_ylabel('Smiling Ranking', fontsize=16, rotation=0, labelpad=20)
    ax.yaxis.set_label_coords(0.1, 0.95)
    
    # X 轴标签
    ax.set_xlabel('time', fontsize=16, loc='right')
    
    # 在曲线末端直接添加文字标注 (模拟原图风格)
    ax.text(x[-1] + 0.2, y_true[-1], 'True Smiling Pattern', color='blue', fontsize=14, va='center')
    ax.text(x[-1] + 0.2, y_not_true[-1], 'Not True Smiling Pattern', color='black', fontsize=14, va='center')

    # 移除刻度数字，保持简洁的手绘感
    ax.set_xticks([])
    ax.set_yticks([])

    # 设置显示范围
    ax.set_xlim(-0.5, 11)
    ax.set_ylim(-1, 10)

    # 显示图像
    plt.tight_layout()
    plt.show()

# 执行函数
if __name__ == "__main__":
    draw_smiling_ranking_graph()