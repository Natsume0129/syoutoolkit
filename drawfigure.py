import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签，如果没有SimHei可以用Arial Unicode MS等
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def draw_smile_vectors_corrected():
    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect('equal')
    
    # 隐藏坐标轴
    ax.axis('off')

    # 定义原点 V0 (Natural)
    origin = np.array([0, 0])

    # --- 定义几何参数 (与原图保持视觉一致) ---
    angle_v2 = 0        # v2 在水平线上
    angle_specific = 22 # specific smile 的角度
    angle_v1 = 48       # v1 (true smile) 的角度

    len_v1 = 11         # v1 长度
    len_v2 = 12         # v2 长度
    len_spec = 9.5      # specific smile 长度

    # 转换弧度
    rad_v2 = np.radians(angle_v2)
    rad_spec = np.radians(angle_specific)
    rad_v1 = np.radians(angle_v1)

    # 计算各端点坐标
    p_v1 = np.array([len_v1 * np.cos(rad_v1), len_v1 * np.sin(rad_v1)])
    p_v2 = np.array([len_v2 * np.cos(rad_v2), len_v2 * np.sin(rad_v2)])
    p_spec = np.array([len_spec * np.cos(rad_spec), len_spec * np.sin(rad_spec)])

    # --- 1. 绘制主向量 ---
    # 绘制 V1 (true smile)
    ax.annotate('', xy=p_v1, xytext=origin, arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(p_v1[0] + 0.3, p_v1[1] + 0.2, r'$v_1$ true smile', fontsize=15, ha='left')

    # 绘制 V2 (polite smile or wry)
    ax.annotate('', xy=p_v2, xytext=origin, arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(p_v2[0] + 0.5, p_v2[1] - 0.2, r'$v_2$ polite smile', fontsize=15, ha='left', va='top')
    ax.text(p_v2[0] + 0.5, p_v2[1] - 1.2, r'or   wry', fontsize=15, ha='left', va='top')

    # 绘制 a specific smile (实线)
    ax.plot([origin[0], p_spec[0]], [origin[1], p_spec[1]], color='black', lw=1.5)
    ax.text(p_spec[0] + 0.8, p_spec[1] + 0.8, 'a specific smile', fontsize=15)

    # 绘制原点标注 V0 Natural
    ax.text(origin[0] - 1.0, origin[1] + 0.8, r'$v_0$', fontsize=15, ha='right')
    ax.text(origin[0] - 0.5, origin[1] - 0.5, 'Netural', fontsize=15, ha='right', va='top')

    # --- 2. 计算并绘制垂线 (虚线 d1 和 d2) ---
    
    # d2: p_spec 在 V2 上的投影
    p_proj_v2 = np.array([p_spec[0], 0])
    ax.plot([p_spec[0], p_proj_v2[0]], [p_spec[1], p_proj_v2[1]], 'k--', lw=1.2)
    ax.text((p_spec[0] + p_proj_v2[0])/2 + 0.3, (p_spec[1] + p_proj_v2[1])/2, r'$d_2$', fontsize=13)

    # d1: p_spec 在 V1 上的投影
    # 投影公式: proj_b(a) = (a·b / |b|^2) * b
    v1_unit = p_v1 / np.linalg.norm(p_v1)
    p_proj_v1 = np.dot(p_spec, v1_unit) * v1_unit
    ax.plot([p_spec[0], p_proj_v1[0]], [p_spec[1], p_proj_v1[1]], 'k--', lw=1.2)
    # d1 文字位置稍微调整以匹配手写图
    ax.text((p_spec[0] + p_proj_v1[0])/2 + 0.2, (p_spec[1] + p_proj_v1[1])/2 + 0.5, r'$d_1$', fontsize=13)

    # --- 3. 绘制直角符号 (修正部分) ---
    s = 0.4 # 直角符号的大小

    # --- d2 的直角 (水平轴上) ---
    # 原图的直角在垂线左侧
    ax.plot([p_proj_v2[0] - s, p_proj_v2[0] - s], [p_proj_v2[1], p_proj_v2[1] + s], 'k', lw=1) # 竖边
    ax.plot([p_proj_v2[0] - s, p_proj_v2[0]], [p_proj_v2[1] + s, p_proj_v2[1] + s], 'k', lw=1) # 横边

    # --- d1 的直角 (旋转轴上) - 核心修正点 ---
    # 我们需要基于投影点 p_proj_v1，沿着 v1 反方向走一点，再沿着 d1 方向走一点，构成正方形
    
    # v1 的单位方向向量
    v1_dir_norm = v1_unit 
    # d1 的单位方向向量 (从投影点指向 specific smile 端点)
    d1_vec = p_spec - p_proj_v1
    d1_dir_norm = d1_vec / np.linalg.norm(d1_vec)
    
    # 计算构成直角符号的另外两个关键点
    # 点 A: 沿着 v1 方向往原点退一步
    pA = p_proj_v1 - s * v1_dir_norm
    # 点 B: 沿着 d1 虚线方向往上走一步
    pB = p_proj_v1 + s * d1_dir_norm
    # 点 C: 构成正方形的角点
    pC = pA + s * d1_dir_norm
    
    # 绘制直角符号的两条边 (A-C 和 C-B)
    ax.plot([pA[0], pC[0]], [pA[1], pC[1]], 'k', lw=1)
    ax.plot([pC[0], pB[0]], [pC[1], pB[1]], 'k', lw=1)


    # --- 4. 绘制角度标注 (theta1 和 theta2) ---
    # theta2 (V2 和 specific smile 之间)
    arc2_rad = 4
    arc2 = Arc(origin, arc2_rad*2, arc2_rad*2, theta1=angle_v2, theta2=angle_specific, color='black', lw=1)
    ax.add_patch(arc2)
    # 计算角度文字坐标
    th2_mid = np.radians((angle_v2 + angle_specific)/2)
    ax.text(arc2_rad * np.cos(th2_mid) + 0.8, arc2_rad * np.sin(th2_mid) - 0.2, r'$\theta_2$', fontsize=13)

    # theta1 (specific smile 和 V1 之间)
    arc1_rad = 4.5
    arc1 = Arc(origin, arc1_rad*2, arc1_rad*2, theta1=angle_specific, theta2=angle_v1, color='black', lw=1)
    ax.add_patch(arc1)
    # 计算角度文字坐标
    th1_mid = np.radians((angle_specific + angle_v1)/2)
    ax.text(arc1_rad * np.cos(th1_mid) + 0.6, arc1_rad * np.sin(th1_mid) + 0.3, r'$\theta_1$', fontsize=13)

    # 设置显示范围以适应所有元素
    ax.set_xlim(-2.5, 15)
    ax.set_ylim(-2.5, 10)

    # 保存或展示
    plt.show()

# 执行绘图
draw_smile_vectors_corrected()