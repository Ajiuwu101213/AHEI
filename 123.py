"""
AHEI MVY v5.5 — 内稳态PPO · 加减速 + 修复报告
================================================
创建者：魏旭  2026.04.30
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import torch
import torch.nn as nn
import torch.optim as optim
import time, math, sys
from datetime import datetime

# ======================== 全局配置 ========================
class Cfg:
    SIZE = 50
    N_FOOD = 20
    FOOD_E = 25.0
    OBS_RAT = 0.03
    INIT_E = 100.0
    TARGET_E = 100.0
    DECAY = 0.2
    VIEW_R = 4
    SMELL_R = 20
    IN_DIM = 3*(2*VIEW_R+1)**2 + 1 + 3
    HID = 128
    ACT_DIM = 5
    MAX_STEP = 5000

    # PPO
    LR = 3e-4
    GAMMA = 0.99
    LAMBDA = 0.95
    CLIP_EPS = 0.2
    ENTROPY_COEF = 0.05
    VALUE_COEF = 0.5
    PPO_EPOCHS = 8
    BATCH_SIZE = 128

    WALL_PENALTY = 0.1          # 撞墙惩罚

    COL_FOOD = np.array([0.2, 0.8, 0.2])
    COL_EMPTY = np.array([0.6, 0.5, 0.35])
    COL_OBS = np.array([0.3, 0.3, 0.3])
    COL_AGENT = np.array([1.0, 0.8, 0.0])

# ======================== 调速器（数字直选 + 加减） ========================
class Speed:
    def __init__(self):
        self.speed = 1.0
        self.paused = False
        self.last_toggle = 0

    def set(self, key):
        if key == "0":
            self.speed = float("inf")
            print("速度：极速")
        elif key.isdigit():
            val = int(key)
            self.speed = 1.0 if val == 1 else val * 5.0
            print(f"速度：{self.speed:.0f}x")
        elif key == '+':
            if self.speed == float("inf"):
                self.speed = 10.0
            else:
                self.speed += 10
            print(f"速度：{self.speed:.0f}x")
        elif key == '-':
            if self.speed == float("inf"):
                self.speed = max(1.0, 10.0)
            else:
                self.speed = max(1.0, self.speed - 10)
            print(f"速度：{self.speed:.0f}x" if self.speed != float("inf") else "速度：极速")

    def toggle(self):
        t = time.time()
        if t - self.last_toggle > 0.3:
            self.paused = not self.paused
            self.last_toggle = t
            print("已暂停" if self.paused else "已恢复")

# ======================== 2D 世界 ========================
class World:
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.sz = Cfg.SIZE
        self.cells = np.full((self.sz, self.sz), 'empty', dtype=object)
        self.food_pos = set()
        self._build()
    def _build(self):
        n_obs = int(self.sz**2 * Cfg.OBS_RAT)
        obs_pos = set()
        while len(obs_pos) < n_obs:
            x, y = np.random.randint(0, self.sz, 2)
            obs_pos.add((x, y))
        for x, y in obs_pos:
            self.cells[x, y] = 'obs'
        self._add_food(Cfg.N_FOOD)
    def _add_food(self, n):
        for _ in range(n):
            while True:
                x, y = np.random.randint(0, self.sz, 2)
                if self.cells[x, y] == 'empty':
                    self.cells[x, y] = 'food'
                    self.food_pos.add((x, y))
                    break
    def eat(self, pos):
        x, y = pos
        if self.cells[x, y] == 'food':
            self.cells[x, y] = 'empty'
            self.food_pos.remove(pos)
            self._add_food(1)
            return Cfg.FOOD_E
        return 0
    def get_rgb_grid(self):
        grid = np.zeros((self.sz, self.sz, 3))
        for i in range(self.sz):
            for j in range(self.sz):
                if self.cells[i, j] == 'food': grid[i, j] = Cfg.COL_FOOD
                elif self.cells[i, j] == 'obs': grid[i, j] = Cfg.COL_OBS
                else: grid[i, j] = Cfg.COL_EMPTY
        return grid
    def view(self, cx, cy):
        r = Cfg.VIEW_R
        v = np.zeros((2*r+1, 2*r+1, 3))
        for i in range(-r, r+1):
            for j in range(-r, r+1):
                x, y = cx+i, cy+j
                if 0 <= x < self.sz and 0 <= y < self.sz:
                    if self.cells[x, y] == 'food': v[i+r, j+r] = Cfg.COL_FOOD
                    elif self.cells[x, y] == 'obs': v[i+r, j+r] = Cfg.COL_OBS
                    else: v[i+r, j+r] = Cfg.COL_EMPTY
                else: v[i+r, j+r] = Cfg.COL_OBS
        return v
    def smell(self, cx, cy):
        if not self.food_pos: return np.array([0,0,0])
        fx, fy = min(self.food_pos, key=lambda p: (p[0]-cx)**2+(p[1]-cy)**2)
        dx, dy = fx-cx, fy-cy
        dist = math.sqrt(dx*dx+dy*dy)
        if dist > Cfg.SMELL_R: return np.array([0,0,0])
        direction = np.array([dx/dist, dy/dist]) if dist>0 else np.array([0,0])
        intensity = max(0, 1.0 - dist/Cfg.SMELL_R)
        return np.array([direction[0], direction[1], intensity])

    def find_safe_birthplace(self):
        """找一个周围至少 5 格都是空地的出生点"""
        for _ in range(500):
            x, y = np.random.randint(0, self.sz, 2)
            if self.cells[x, y] != 'empty':
                continue
            empty_neighbors = 0
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < self.sz and 0 <= ny < self.sz:
                        if self.cells[nx, ny] == 'empty':
                            empty_neighbors += 1
            if empty_neighbors >= 5:
                return x, y
        while True:
            x, y = np.random.randint(0, self.sz, 2)
            if self.cells[x, y] == 'empty':
                return x, y

# ======================== PPO 网络 ========================
class Brain(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(Cfg.IN_DIM, Cfg.HID), nn.ReLU(),
            nn.Linear(Cfg.HID, Cfg.HID//2), nn.ReLU()
        )
        self.actor = nn.Sequential(nn.Linear(Cfg.HID//2, Cfg.ACT_DIM), nn.Softmax(-1))
        self.critic = nn.Linear(Cfg.HID//2, 1)
    def forward(self, x):
        e = self.enc(x)
        return self.actor(e), self.critic(e)
    def act(self, x):
        actp, val = self.forward(x)
        dist = torch.distributions.Categorical(actp)
        action = dist.sample()
        return action, dist.log_prob(action), val.squeeze(-1)
    def evaluate(self, states, actions):
        actp, val = self.forward(states)
        dist = torch.distributions.Categorical(actp)
        return dist.log_prob(actions), dist.entropy(), val.squeeze(-1)

# ======================== PPO Buffer ========================
class PPOBuffer:
    def __init__(self):
        self.states, self.actions, self.log_probs = [], [], []
        self.rewards, self.values, self.dones = [], [], []
    def store(self, s, a, lp, r, v, d):
        self.states.append(s); self.actions.append(a)
        self.log_probs.append(lp); self.rewards.append(r)
        self.values.append(v); self.dones.append(d)
    def clear(self):
        self.__init__()
    def compute_gae(self, last_val=0):
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_val])
        dones = np.array(self.dones + [0])
        gae = 0
        advantages = np.zeros_like(rewards)
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + Cfg.GAMMA*values[t+1]*(1-dones[t]) - values[t]
            gae = delta + Cfg.GAMMA*Cfg.LAMBDA*(1-dones[t])*gae
            advantages[t] = gae
        returns = advantages + values[:-1]
        return torch.FloatTensor(advantages), torch.FloatTensor(returns)

# ======================== 智能体 ========================
class Agent:
    def __init__(self, world):
        self.w = world
        self.x, self.y = world.find_safe_birthplace()
        self.energy = Cfg.INIT_E
        self.alive = True
        self.eaten = 0
        self.prev_drive = self._compute_drive()
    def _compute_drive(self):
        return (self.energy / Cfg.TARGET_E - 1.0) ** 2
    def sense(self):
        v = self.w.view(self.x, self.y).flatten()
        e = np.array([self.energy / Cfg.INIT_E])
        s = self.w.smell(self.x, self.y)
        return np.concatenate([v, e, s])
    def move(self, a):
        dx, dy = 0, 0
        if a == 0: dx = -1
        elif a == 1: dx = 1
        elif a == 2: dy = -1
        elif a == 3: dy = 1
        nx, ny = self.x+dx, self.y+dy
        if 0 <= nx < self.w.sz and 0 <= ny < self.w.sz and self.w.cells[nx, ny] != 'obs':
            self.x, self.y = nx, ny
            return True, 0.0
        return False, Cfg.WALL_PENALTY

    def step(self, a):
        moved, wall_penalty = self.move(a)
        gain = self.w.eat((self.x, self.y))
        if gain > 0:
            self.energy += gain
            self.eaten += 1
        self.energy -= Cfg.DECAY
        if self.energy <= 0:
            self.energy = 0
            self.alive = False
        current_drive = self._compute_drive()
        reward = (self.prev_drive - current_drive) - wall_penalty
        self.prev_drive = current_drive
        return reward

# ======================== 训练器 ========================
class Trainer:
    def __init__(self, speed):
        self.sp = speed
        self.world = World()
        self.agent = Agent(self.world)
        self.net = Brain()
        self.opt = optim.Adam(self.net.parameters(), lr=Cfg.LR)
        self.buffer = PPOBuffer()
        self.food_hist, self.surv_hist = [], []
        self.reward_hist, self.drive_hist, self.energy_hist = [], [], []
        self.ep = self.step_cnt = 0
        self.quit = False
        self._init_plot()

    def _int_fmt(self, x, p): return f'{int(x)}'

    def _init_plot(self):
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei','SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.ion()
        self.fig,(self.ax1,self.ax2) = plt.subplots(1,2,figsize=(12,5))
        self.fig.canvas.manager.set_window_title("AHEI MVY v5.5 加减速")
        plt.subplots_adjust(left=0.06, right=0.94, bottom=0.12, top=0.9, wspace=0.3)
        self.ax2b = self.ax2.twinx()
        self.ax2b.set_ylabel('存活步数', color='red', fontsize=10)
        self.ax2.set_xlabel('训练轮次', fontsize=10)
        self.ax2.set_ylabel('觅食次数', color='blue', fontsize=10)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _on_key(self, e):
        if e.key == 'q': self.quit = True
        elif e.key == ' ': self.sp.toggle()
        elif e.key in ['+', '-']: self.sp.set(e.key)
        elif e.key.isdigit(): self.sp.set(e.key)

    def _draw(self):
        if self.sp.speed == float("inf") or self.sp.paused: return
        rf = 1 if self.sp.speed <= 1 else (2 if self.sp.speed <= 3 else (5 if self.sp.speed <= 5 else 10))
        if self.step_cnt % rf != 0: return

        self.ax1.clear()
        rgb = self.world.get_rgb_grid()
        self.ax1.imshow(rgb, origin='upper')
        self.ax1.scatter(self.agent.y, self.agent.x, color=Cfg.COL_AGENT, s=120, edgecolors='black', linewidth=1.5, zorder=10)
        self.ax1.set_title(f"轮 {self.ep} | 能量 {self.agent.energy:.0f} | 已吃 {self.agent.eaten}")
        speed_str = "极速" if self.sp.speed==float("inf") else f"{self.sp.speed:.0f}x"
        self.ax1.set_xlabel(f"速度:{speed_str} 暂停:{'是' if self.sp.paused else '否'}")

        self.ax2.clear(); self.ax2b.clear()
        if self.food_hist:
            smooth = max(1, min(30, len(self.food_hist)//2))
            f = np.array(self.food_hist); s = np.array(self.surv_hist)
            if len(f) >= smooth:
                f_s = np.convolve(f, np.ones(smooth)/smooth, mode='valid')
                s_s = np.convolve(s, np.ones(smooth)/smooth, mode='valid')
            else:
                f_s, s_s = f, s
            self.ax2.plot(f, alpha=0.2, color='blue', lw=0.5)
            self.ax2.plot(range(len(f_s)), f_s, 'b-', lw=1.5, label='觅食')
            self.ax2.set_ylabel('觅食次数', color='blue', fontsize=10)
            self.ax2.yaxis.set_major_formatter(FuncFormatter(self._int_fmt))
            max_f = max(f) if len(f)>0 else 1
            self.ax2.set_ylim(-0.5, max_f + 1.5)

            self.ax2b.plot(s, alpha=0.2, color='red', lw=0.5)
            self.ax2b.plot(range(len(s_s)), s_s, 'r-', lw=1.5, label='存活')
            self.ax2b.set_ylabel('存活步数', color='red', fontsize=10)
            self.ax2b.yaxis.set_major_formatter(FuncFormatter(self._int_fmt))
            max_s = max(s) if len(s)>0 else 1
            self.ax2b.set_ylim(-50, max_s + 200)

            self.ax2.set_title(f'进度 (第 {self.ep} 轮)')
            self.ax2.set_xlabel('训练轮次')
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def _ppu_update(self):
        states = torch.FloatTensor(np.array(self.buffer.states))
        actions = torch.LongTensor(self.buffer.actions)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs)

        with torch.no_grad():
            s_last = torch.FloatTensor(self.agent.sense()).unsqueeze(0)
            _, last_val = self.net.forward(s_last)
            last_val = last_val.item() if self.agent.alive else 0
        advantages, returns = self.buffer.compute_gae(last_val)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_steps = len(states)
        for _ in range(Cfg.PPO_EPOCHS):
            indices = np.random.permutation(total_steps)
            for start in range(0, total_steps, Cfg.BATCH_SIZE):
                batch_idx = indices[start:start+Cfg.BATCH_SIZE]
                s_batch = states[batch_idx]; a_batch = actions[batch_idx]
                old_lp_batch = old_log_probs[batch_idx]
                adv_batch = advantages[batch_idx]; ret_batch = returns[batch_idx]

                new_lp, entropy, values = self.net.evaluate(s_batch, a_batch)
                ratio = torch.exp(new_lp - old_lp_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1-Cfg.CLIP_EPS, 1+Cfg.CLIP_EPS) * adv_batch
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (ret_batch - values).pow(2).mean()
                loss = actor_loss + Cfg.VALUE_COEF*critic_loss - Cfg.ENTROPY_COEF*entropy.mean()
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.opt.step()
        self.buffer.clear()

    def run(self):
        print("="*50)
        print("AHEI MVY v5.5 — 撞墙惩罚 + 加减速")
        print("1-9: 直选倍速 | +/-: 加减10倍速 | 0: 极速")
        print("="*50)
        ep = 0
        try:
            while not self.quit:
                ep += 1; self.ep = ep
                self.agent = Agent(self.world)
                step = 0
                total_reward, total_drive, total_energy = 0.0, 0.0, 0.0

                while self.agent.alive and step < Cfg.MAX_STEP and not self.quit:
                    while self.sp.paused and not self.quit:
                        self.fig.canvas.flush_events()
                        time.sleep(0.05)
                    if self.quit: break

                    s = torch.FloatTensor(self.agent.sense()).unsqueeze(0)
                    action, log_prob, value = self.net.act(s)
                    old_state = s.numpy().flatten()
                    reward = self.agent.step(action.item())
                    done = not self.agent.alive

                    self.buffer.store(old_state, action.item(), log_prob.item(),
                                      reward, value.item(), done)
                    step += 1; self.step_cnt += 1
                    total_reward += reward
                    total_drive += self.agent.prev_drive
                    total_energy += self.agent.energy

                    if self.sp.speed == float("inf"):
                        if step % 1000 == 0: self.fig.canvas.flush_events()
                    else:
                        self._draw()
                        self.fig.canvas.flush_events()

                if self.quit: break
                self._ppu_update()

                self.food_hist.append(self.agent.eaten)
                self.surv_hist.append(step)
                self.reward_hist.append(total_reward)
                self.drive_hist.append(total_drive / max(1, step))
                self.energy_hist.append(total_energy / max(1, step))

                if ep % 50 == 0 and len(self.food_hist)>=50:
                    print(f"轮 {ep:5d} | 均觅食 {np.mean(self.food_hist[-50:]):.1f} | 均存活 {np.mean(self.surv_hist[-50:]):.0f}")
        finally:
            self._gen_report()
        plt.ioff()
        plt.show()

    def _gen_report(self):
        if len(self.food_hist) < 50:
            print("轮次不足50，未生成报告")
            return

        food_arr = np.array(self.food_hist)
        surv_arr = np.array(self.surv_hist)
        reward_arr = np.array(self.reward_hist)
        drive_arr = np.array(self.drive_hist)
        energy_arr = np.array(self.energy_hist)

        total_episodes = len(food_arr)  # 使用实际数组长度
        n_segments = min(10, total_episodes // 100) if total_episodes >= 200 else max(1, total_episodes // 50)
        seg_size = max(1, total_episodes // n_segments)
        segments = []
        for i in range(n_segments):
            start = i * seg_size
            end = (i+1) * seg_size if i < n_segments-1 else total_episodes
            seg = slice(start, end)
            segments.append({
                "range": f"{start+1}-{end}",
                "mean_food": np.mean(food_arr[seg]),
                "std_food": np.std(food_arr[seg]),
                "max_food": np.max(food_arr[seg]),
                "mean_surv": np.mean(surv_arr[seg]),
                "std_surv": np.std(surv_arr[seg]),
                "max_surv": np.max(surv_arr[seg]),
                "mean_reward": np.mean(reward_arr[seg]),
                "mean_drive": np.mean(drive_arr[seg]),
                "mean_energy": np.mean(energy_arr[seg]),
            })

        a1 = np.mean(food_arr[:50]) if total_episodes >= 50 else np.mean(food_arr)
        a2 = np.mean(food_arr[-50:]) if total_episodes >= 50 else np.mean(food_arr)
        improvement = a2 - a1

        x = np.arange(total_episodes)
        food_slope = np.polyfit(x, food_arr, 1)[0] if total_episodes >= 2 else 0
        surv_slope = np.polyfit(x, surv_arr, 1)[0] if total_episodes >= 2 else 0
        drive_slope = np.polyfit(x, drive_arr, 1)[0] if total_episodes >= 2 else 0

        verdict = "✅ 核心闭环验证成功" if improvement > 3 else ("📈 微弱信号" if improvement > 1 else "❌ 未检测到显著学习")

        seg_table = ""
        for seg in segments:
            seg_table += f"  {seg['range']:>8s} | {seg['mean_food']:6.2f} | {seg['std_food']:6.2f} | {seg['max_food']:5.0f} | {seg['mean_surv']:7.0f} | {seg['std_surv']:7.0f} | {seg['max_surv']:6.0f} | {seg['mean_reward']:8.4f} | {seg['mean_drive']:8.4f} | {seg['mean_energy']:7.1f}\n"

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = f"experiment_report_{ts}.txt"
        report = f"""================================================================================
                    AHEI MVY 实验报告 (v5.5 — 加减速)
                    创建者：魏旭  时间：{ts}
================================================================================
世界: {Cfg.SIZE}x{Cfg.SIZE}  食物: {Cfg.N_FOOD}  障碍: {Cfg.OBS_RAT*100:.1f}%
能量: 初始{Cfg.INIT_E} 衰减{Cfg.DECAY}/步 进食+{Cfg.FOOD_E} 目标{Cfg.TARGET_E}
感官: 视野{Cfg.VIEW_R} 嗅觉{Cfg.SMELL_R}
算法: PPO (γ={Cfg.GAMMA}, λ={Cfg.LAMBDA}, clip={Cfg.CLIP_EPS})
奖励: 驱力下降 - 撞墙惩罚({Cfg.WALL_PENALTY})
总轮数: {total_episodes}  LR: {Cfg.LR}  熵系数: {Cfg.ENTROPY_COEF}
------------------------------------------------
前50均觅食: {a1:.2f}  后50均觅食: {a2:.2f}  提升: {improvement:+.2f}
趋势斜率: 觅食{food_slope:+.4f}/轮, 存活{surv_slope:+.1f}/轮, 驱力{drive_slope:+.4f}/轮
最高单轮觅食: {np.max(food_arr)}  最长存活: {np.max(surv_arr)}
{seg_table}
结论: {verdict}
================================================================================
"""
        with open(fn, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n详细实验报告已生成：{fn}")

if __name__ == "__main__":
    print("\nAHEI MVY v5.5 — 内稳态驱动PPO\n创建者：魏旭\n")
    sc = Speed()
    tr = Trainer(sc)
    tr.run()