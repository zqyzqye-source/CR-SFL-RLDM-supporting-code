import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.stats as stats
import math
import warnings

# 忽略积分收敛警告
warnings.filterwarnings("ignore")

# ==========================================
# 0. IEEE 顶刊图表全局字体设置 (Times New Roman)
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'  # 确保公式中的希腊字母 (\sigma, \lambda) 也是 Times 风格

# ==========================================
# 1. 基础系统与实验全局参数
# ==========================================
N = 10000          # 迭代步数
trials = 20        # 保持20次蒙特卡洛，这是展示毛刺感的黄金次数
x_range = 6.0      
M=3
T_censor = 1.0     # 截断阈值 
p_outlier = 0.02   # 脉冲发生概率
sigma_I = 10.0     # 脉冲噪声标准差

# 【核心修正】：废除 0.9991 的过度平滑！理论与仿真统一使用真实的 0.995
LAMBDA_VAL = 0.995 

Q = 31             
delta_x = (2 * x_range) / (Q - 3)  

C_matrix = 0.5 * np.array([[-1,  3, -3,  1],
                           [ 2, -5,  4, -1],
                           [-1,  0,  1,  0],
                           [ 0,  2,  0,  0]])

var_Gs = [0.01, 0.05, 0.10]
colors = ['blue', 'black', 'red']  

# ==========================================
# 2. 构建拓扑与 CR样条活跃子空间严格剥离
# ==========================================
def get_spline_feature(x):
    x_clip = np.clip(x, -x_range + 1e-5, x_range - 1e-5)
    i = int(np.floor((x_clip + x_range) / delta_x))
    i = max(0, min(i, Q - 4))
    u = (x_clip + x_range) / delta_x - i
    Phi = np.zeros(Q)
    Phi[i:i+4] = C_matrix.T @ np.array([u**3, u**2, u, 1.0])
    return Phi 

print(">> 正在计算协方差矩阵并评估 CR 样条节点激励情况...")
R_mat = np.zeros((Q, Q))
x_dense = np.linspace(-x_range, x_range, 20000)
for xv in x_dense:
    feat = get_spline_feature(xv)
    R_mat += np.outer(feat, feat)
R_mat /= len(x_dense)

# 提取活跃子空间：只对得到充分激励的 CR 样条节点进行求逆
diag_R = np.diag(R_mat)
active_mask = diag_R > (np.max(diag_R) * 0.01)
active_idx = np.where(active_mask)[0]
null_idx = np.where(~active_mask)[0]

R_act = R_mat[np.ix_(active_idx, active_idx)]
Tr_R_inv = np.trace(np.linalg.inv(R_act))
print(f"✅ 有效活跃节点: {len(active_idx)}/{Q}, 理论稳态迹 Tr(R^-1): {Tr_R_inv:.2f}")

np.random.seed(42)
w_0_raw = np.random.randn(Q) * 0.5  
w_0 = np.zeros(Q)
w_0[active_idx] = w_0_raw[active_idx]

# ==========================================
# 3. 完美数值积分理论界限 (纯数学，无人工系数)
# ==========================================
def exact_bg_cr(y_est, d_obs, T, p, sig_G, sig_I):
    eps = 1e-5
    if d_obs >= T - eps:
        c = T - y_est
        x_G, x_I = np.clip(c / sig_G, -20.0, 20.0), np.clip(c / sig_I, -20.0, 20.0)
        num = (1 - p) * sig_G * stats.norm.pdf(x_G) + p * sig_I * stats.norm.pdf(x_I)
        den = (1 - p) * 0.5 * math.erfc(x_G / math.sqrt(2)) + p * 0.5 * math.erfc(x_I / math.sqrt(2)) + 1e-250
        return y_est + num / den
    elif d_obs <= -T + eps:
        c = -T - y_est
        x_G, x_I = np.clip(c / sig_G, -20.0, 20.0), np.clip(c / sig_I, -20.0, 20.0)
        num = -(1 - p) * sig_G * stats.norm.pdf(x_G) - p * sig_I * stats.norm.pdf(x_I)
        den = (1 - p) * 0.5 * math.erfc(-x_G / math.sqrt(2)) + p * 0.5 * math.erfc(-x_I / math.sqrt(2)) + 1e-250
        return y_est + num / den
    return d_obs

def get_ldm_weight(err, sig_L=1.0):
    z = np.clip(err / sig_L, -20.0, 20.0)
    if abs(z) < 1e-5: return 1.0 
    num = math.exp(z) - math.exp(-z)
    den = err * (sig_L**2) * ((math.exp(z) + 2.0 + math.exp(-z))**2)
    weight = (num / den) * (8.0 * sig_L**3)
    return max(min(weight, 1.0), 0.0)

def compute_theoretical_limit(var_G):
    sig_G = math.sqrt(var_G)
    def pdf_mix(v):
        return (1 - p_outlier)*stats.norm.pdf(v,0,sig_G) + p_outlier*stats.norm.pdf(v,0,sigma_I)

    # 抽样计算期望
    y_samples = [np.dot(w_0, get_spline_feature(x)) for x in np.linspace(-x_range, x_range, 30)]
    
    var_list, g_list = [], []
    for y in y_samples:
        print(".", end="", flush=True)
        # 算方差 E[e^2 * g^2]
        def integrand_var(v):
            d_obs = np.clip(y + v, -T_censor, T_censor)
            e_n = exact_bg_cr(y, d_obs, T_censor, p_outlier, sig_G, sigma_I) - y
            w_ldm = get_ldm_weight(e_n, 1.0) # 理论核与仿真统一
            return (w_ldm**2) * (e_n**2) * pdf_mix(v)
        v_var, _ = integrate.quad(integrand_var, -20, 20, limit=50)
        var_list.append(v_var)
        
        # 算期望 E[g]
        def integrand_g(v):
            d_obs = np.clip(y + v, -T_censor, T_censor)
            e_n = exact_bg_cr(y, d_obs, T_censor, p_outlier, sig_G, sigma_I) - y
            return get_ldm_weight(e_n, 1.0) * pdf_mix(v)
        v_g, _ = integrate.quad(integrand_g, -20, 20, limit=50)
        g_list.append(v_g)

    E_var = np.mean(var_list)
    E_g = np.mean(g_list)
    
    Theory_val = 0.5 * (1 - LAMBDA_VAL) * E_var / (E_g**2) * Tr_R_inv
    return 10 * np.log10(Theory_val + 1e-15)

print(">> 正在精确计算无补偿系数的纯正理论界限:")
theory_lines = []
for vg in var_Gs:
    print(f"\n   计算 var={vg:.2f} ", end="")
    val = compute_theoretical_limit(vg)
    theory_lines.append(val)
    print(f" 完成! [理论: {val:.2f} dB]")

# ==========================================
# 4. 蒙特卡洛仿真 (保留原汁原味的毛刺)
# ==========================================
sim_curves = []
print("\n>> 开始运行自适应滤波蒙特卡洛仿真...")
for idx, var_G in enumerate(var_Gs):
    sig_G = math.sqrt(var_G)
    error_accum = np.zeros(N)
    
    for trial in range(trials):
        np.random.seed(trial + idx * 888)
        w = np.zeros(Q)
        P = np.eye(Q) * 50.0  
        X = np.random.uniform(-x_range, x_range, N)
        
        for n in range(N):
            x = X[n]
            feat = get_spline_feature(x)
            y = np.dot(w_0, feat)
            y_est = np.dot(w, feat)
            
            error_accum[n] += np.sum((w_0 - w)**2)
            
            noise_G = np.random.normal(0, sig_G)
            noise_I = np.random.normal(0, sigma_I) if np.random.rand() < p_outlier else 0.0
            d_obs = np.clip(y + noise_G + noise_I, -T_censor, T_censor)
            
            d_bar = exact_bg_cr(y, d_obs, T_censor, p_outlier, sig_G, sigma_I)
            e_n = d_bar - y_est
            g_val = get_ldm_weight(e_n, 1.0)
            
            P_h = P @ feat
            denom = LAMBDA_VAL + g_val * np.dot(feat, P_h)
            k_n = (g_val * P_h) / max(denom, 1e-12)
            
            w += k_n * e_n
            
            w[null_idx] = 0.0
            
            P = (P - np.outer(k_n, P_h)) / LAMBDA_VAL
            P = (P + P.T) / 2.0
            
            if len(null_idx) > 0:
                P[null_idx, :] = 0.0
                P[:, null_idx] = 0.0
                P[null_idx, null_idx] = 50.0

    error_db = 10 * np.log10(error_accum / trials + 1e-12)
    sim_curves.append(error_db)
    print(f"   -> 方差 {var_G} 仿真完成。")

# ==========================================
# 5. 顶刊视觉出图
# ==========================================
# 【完全保留您原有的 figsize 和 fontsize 设置】
plt.figure(figsize=(8, 5.8), dpi=200)
x_axis = np.arange(N)

for i in range(3):
    var_G = var_Gs[i]
    color = colors[i]
    label_sim = rf'Simulation ($\sigma_1^2={var_G:.2f}, \lambda={LAMBDA_VAL}, \sigma_L=0.5$)'
    label_theory = rf'Theory ($\sigma_1^2={var_G:.2f}, \lambda={LAMBDA_VAL}, \sigma_L=0.5$)'
    
    plt.plot(x_axis, sim_curves[i], color=color, linewidth=1.5, alpha=0.9, label=label_sim)
    plt.axhline(y=theory_lines[i], color=color, linewidth=2.0, linestyle='--', label=label_theory)

plt.xlabel('Iteration', fontsize=28)
plt.ylabel('MSD (dB)', fontsize=28)
plt.xlim([0, N])
plt.ylim([-20, 10]) 

plt.grid(True, which="both", ls="--", alpha=0.5, color='gray')
plt.tick_params(axis='both', labelsize=28)

# 图例大小和样式完全不动，仅受全局 Times New Roman 影响
plt.legend(fontsize=15, loc='upper right', framealpha=1.0, edgecolor='black', shadow=True, ncol=1)

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)

save_folder = r"C:\Users\lenovo\Desktop\Claude\tupian" 
os.makedirs(save_folder, exist_ok=True)
save_path = os.path.join(save_folder, "fig_A1_MSD_sigma.pdf")
plt.savefig(save_path, format='pdf', bbox_inches='tight')
print(f"\n✅ 图片已成功保存至: {save_path}")
plt.show()