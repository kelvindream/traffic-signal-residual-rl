"""CoLight 基线（E12 预注册修订 1）：多头图注意力交通信号控制（Wei et al., CIKM 2019）。

改编基线（预注册第 5 节"实现适配预声明" + 修订 1"CoLight 实现口径"），
以下适配决策全部显式记录，不隐藏差异：

A1 联合图输入 vs 独立 learner 外壳：
    原方法对所有路口做参数共享的图注意力；本管线（train_eval_multi_baseline.py）
    以"每路口一个独立 learner"的接口调用，run_episode 只向 learner.act 传本地
    48-D 观测。为此 CoLightLearner 做成薄包装：4 个实例共享同一个 CoLightEngine
    （单一共享网络 + 共享回放缓冲，由 build_learners 特例装配），引擎在每次 act
    时把该路口最新观测写入共享节点缓存，用 [4, 48] 联合观测前向，取本路口行的
    Q 值。忠实保留 CoLight 的跨路口注意力与参数共享，是对管线外壳的最小适配。

A2 图拓扑：全互联注意力（4 节点两两可见，含自环）。
    预注册允许"按真实邻接或全互联"。2×2 小图每个路口真实邻居仅 2 个，且需解析
    SUMO net 拓扑引入额外脆弱性；全互联是其超集（注意力权重可学到等价的稀疏
    结构），选择全互联避免硬编码拓扑假设。

A3 输入嵌入：48-D 局部观测（32 原始 + 16 增强，与管线学习基线同口径）→
    输入 LayerNorm（P4 工程修复，同 BasicDqnNetwork/SAC 系网络）→ 小 MLP 嵌入
    （48→64→32，ReLU）。原论文为单层 Dense 嵌入，此处按任务口径用小 MLP。

A4 图注意力骨干：num_layers=2 层多头图注意力（heads=4，预注册口径；
    embed_dim=32，head_dim=8），每层 nn.MultiheadAttention(Q=K=V=节点嵌入) +
    残差 + ReLU。

A5 输出头：每节点二元动作 Q 值 {HOLD=0, TERMINATE=1}（原方法输出单标量相位
    Q，本环境为二元动作空间，输出头 32→64→2）。

A6 学习规则：DQN 式 TD（MSE），目标网络软更新。
    - lr=1e-3、gamma=0.99：预注册指定（CoLight 原论文口径）；
    - batch_size=128、buffer_size=100_000、tau=0.005、learn_frequency=4：
      沿用管线 off-policy 统一口径（与 SAC 一致）；learn_frequency 以引擎
      update() 调用次数计，4 个路口每 env.step 各调用一次 → 等效每 env.step
      一次梯度更新，与 SAC 的有效更新频率一致；
    - ε-greedy 探索（DQN 标准做法，原论文口径）：ε 1.0→0.05 指数退火，
      eps_decay 以引擎 act() 调用次数计（约 4 路口 × 60 决策步/回合 ×
      300 回合 ≈ 72k 次，decay=20000 使前半程充分探索、训练后段退火至低
      探索率，适配本环境 300-episode 预算）；
    - 目标网络用软更新（tau=0.005）替代原论文周期性硬更新：与管线
      SAC/HDRL 的目标网络口径统一。

A7 动作掩码：off-policy 口径与 SAC/HDRL 完全一致——act 返回原始（可能非法）
    动作，由管线 apply_mask 覆盖为合法动作，回放存执行动作（invalid 动作率恒 0）。

A8 联合转移的回放组装：管线对每个 acted 路口分别调用 observe(本地 s, a_i,
    本地 s', r_i, done)，引擎延迟一步组装联合转移：observe 只登记
    (节点 i, a_i, r_i, done) 与该路口下一观测缓存；在下一决策步首次 act 时
    （或回合末 end_episode）用此刻的联合下一观测 [4,48] 将
    (联合 s, 节点 i, a_i, 联合 s', r_i, done) 落入回放缓冲。done=1 的尾部
    转移其 s' 无关紧要（bootstrap 被 done 屏蔽）。若某路口某步未 ready，
    其缓存观测沿用最近一次已知值（图输入允许非活跃路口 stale，改编决策）；
    评估回合跨回合首步可能残留上一回合缓存观测，仅影响该步的邻居输入
    （均为已接受的改编近似）。

A9 checkpoint：save() 把共享引擎完整状态写入管线要求的 4 个
    learner_{i}_{tls}.pt（内容相同，幂等）；load() 顺序加载 4 次，内容一致
    故等价于加载一次；--resume 与冻结评估机制不受影响。回放缓冲不持久化，
    与其他基线的 resume 行为一致。
"""

import math
import random
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 与 agents/basic_dqn.py、experiments/agent_factory.py 同口径的设备约定
device = "cuda" if torch.cuda.is_available() else "cpu"

# 回放元组：联合观测 [num_nodes, obs_dim]、决策节点、执行动作、联合下一观测、奖励、done
JointTransition = namedtuple(
    "JointTransition", ("state", "node", "action", "next_state", "reward", "done"))


def _apply_mask(action, mask):
    """与 train_eval_multi_baseline.apply_mask 同口径（避免 tools→agents 环状导入）。"""
    action = int(action)
    if mask is not None and not bool(mask[action]):
        action = 1 if bool(mask[1]) else 0
    return action


def _init_linear(layer, gain):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


class CoLightNetwork(nn.Module):
    """CoLight 骨干：输入嵌入（小 MLP）→ 多头图注意力 → 每节点二元 Q 头（A3/A4/A5）。"""

    def __init__(self, input_dim, action_dim, embed_dim=32, heads=4, num_layers=2):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.embed = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, embed_dim), nn.ReLU(),
        )
        _init_linear(self.embed[0], 2 ** 0.5)
        _init_linear(self.embed[2], 2 ** 0.5)
        # 全互联图注意力（A2）：4 节点 token 两两注意力，heads=4（预注册口径）
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(),
            nn.Linear(64, action_dim),
        )
        _init_linear(self.head[0], 2 ** 0.5)
        _init_linear(self.head[2], 1.0)

    def forward(self, x):
        """x: [B, num_nodes, input_dim] → Q: [B, num_nodes, action_dim]"""
        h = self.embed(self.input_norm(x))
        for attention in self.attention_layers:
            attn_out, _ = attention(h, h, h, need_weights=False)
            h = F.relu(h + attn_out)  # 残差 + ReLU
        return self.head(h)


class CoLightEngine:
    """共享 CoLight 引擎：单一参数共享网络 + 共享回放缓冲，4 个路口包装器共用（A1）。"""

    def __init__(self, params, input_dim, action_dim, num_nodes=4):
        self.params = dict(params)
        self.action_dim = int(action_dim)
        self.num_nodes = int(num_nodes)
        self.gamma = float(params["gamma"])
        self.batch_size = int(params["batch_size"])
        self.buffer_size = int(params["buffer_size"])
        self.tau = float(params["tau"])
        self.learn_frequency = int(params["learn_frequency"])
        self.eps_start = float(params["eps_start"])
        self.eps_end = float(params["eps_end"])
        self.eps_decay = float(params["eps_decay"])
        net_kwargs = dict(
            embed_dim=int(params.get("embed_dim", 32)),
            heads=int(params.get("heads", 4)),
            num_layers=int(params.get("num_layers", 2)),
        )
        self.network = CoLightNetwork(input_dim, action_dim, **net_kwargs).to(device)
        self.target_network = CoLightNetwork(input_dim, action_dim, **net_kwargs).to(device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=float(params["lr"]))
        self.memory = []  # JointTransition 环形缓冲
        self._position = 0
        zeros = np.zeros(input_dim, dtype=np.float32)
        # 节点观测缓存：act 写 cache，observe 写 next_cache（A8）
        self.cache = [zeros.copy() for _ in range(self.num_nodes)]
        self.next_cache = [zeros.copy() for _ in range(self.num_nodes)]
        self.pending = []  # 已 observe、待联合下一观测完整后落盘的转移
        self.node_steps = 0  # 训练 act() 调用计数（ε 退火用）
        self.update_calls = 0
        self.learn_steps = 0

    # ---- 决策 ----
    def act(self, node_index, state, evaluate=False):
        state = np.asarray(state, dtype=np.float32)
        if self.pending:
            # 上一决策步的转移此刻联合下一观测已完整，落盘（A8）
            self._flush_pending()
        self.cache[node_index] = state.copy()
        joint = torch.from_numpy(np.stack(self.cache)).unsqueeze(0).to(device)
        with torch.no_grad():
            q_row = self.network(joint)[0, node_index]
        if evaluate:
            return int(torch.argmax(q_row).item())
        self.node_steps += 1
        eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -self.node_steps / self.eps_decay)
        if random.random() < eps:
            return random.randrange(self.action_dim)
        return int(torch.argmax(q_row).item())

    # ---- 经验 ----
    def observe(self, node_index, state, action, next_state, reward, done):
        self.next_cache[node_index] = np.asarray(next_state, dtype=np.float32).copy()
        joint_state = np.stack(self.cache).astype(np.float32)  # 本步联合观测快照
        self.pending.append(
            (int(node_index), int(action), float(reward), float(done), joint_state))

    def _flush_pending(self):
        joint_next = np.stack(self.next_cache).astype(np.float32)
        for node, action, reward, done, joint_state in self.pending:
            self._push(JointTransition(joint_state, node, action,
                                       joint_next, reward, done))
        self.pending.clear()
        zeros = np.zeros_like(self.next_cache[0])
        self.next_cache = [zeros.copy() for _ in range(self.num_nodes)]

    def _push(self, transition):
        if len(self.memory) < self.buffer_size:
            self.memory.append(transition)
        else:
            self.memory[self._position] = transition
        self._position = (self._position + 1) % self.buffer_size

    # ---- 学习 ----
    def update(self):
        self.update_calls += 1
        if len(self.memory) < self.batch_size:
            return None
        if self.update_calls % self.learn_frequency != 0:
            return None
        batch = random.sample(self.memory, self.batch_size)
        rows = torch.arange(self.batch_size, device=device)
        states = torch.from_numpy(
            np.stack([t.state for t in batch])).to(device)
        next_states = torch.from_numpy(
            np.stack([t.next_state for t in batch])).to(device)
        nodes = torch.tensor([t.node for t in batch], dtype=torch.long, device=device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device)

        q_all = self.network(states)  # [B, N, A]
        q_sa = q_all[rows, nodes].gather(1, actions.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            next_all = self.target_network(next_states)  # [B, N, A]
            next_q = next_all[rows, nodes].max(dim=1).values
        target = rewards + self.gamma * (1.0 - dones) * next_q
        loss = F.mse_loss(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # 目标网络软更新（A6：管线 tau 口径，替代原论文周期性硬更新）
        with torch.no_grad():
            for target_param, param in zip(self.target_network.parameters(),
                                           self.network.parameters()):
                target_param.mul_(1.0 - self.tau).add_(self.tau * param)
        self.learn_steps += 1
        return {"critic_loss": float(loss.item())}

    def end_episode(self):
        """回合末落盘剩余 pending 转移；被 4 个包装器各调用一次，幂等。"""
        if self.pending:
            self._flush_pending()
        return None

    def get_stats(self):
        return {"learn_steps": int(self.learn_steps),
                "replay_size": len(self.memory)}

    # ---- 持久化 ----
    def save(self, path):
        torch.save({
            "network": self.network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "node_steps": self.node_steps,
            "update_calls": self.update_calls,
            "learn_steps": self.learn_steps,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=device)
        self.network.load_state_dict(ckpt["network"])
        self.target_network.load_state_dict(ckpt["target_network"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.node_steps = int(ckpt.get("node_steps", 0))
        self.update_calls = int(ckpt.get("update_calls", 0))
        self.learn_steps = int(ckpt.get("learn_steps", 0))


class CoLightLearner:
    """管线 learner 接口的薄包装（A1）：4 个实例共享同一 CoLightEngine。

    接口与 train_eval_multi_baseline.SacLearner 等完全一致：
    act / observe / update / end_episode / get_stats / save / load。
    """

    def __init__(self, params, input_dim, output_dim, mode="train",
                 engine=None, node_index=0, num_nodes=4):
        if engine is None:
            engine = CoLightEngine(params, input_dim, output_dim, num_nodes=num_nodes)
        self.engine = engine
        self.node_index = int(node_index)
        self.mode = mode

    def act(self, state, evaluate=False, mask=None):
        return _apply_mask(
            self.engine.act(self.node_index, state, evaluate=evaluate), mask)

    def observe(self, state, action, next_state, reward, done):
        self.engine.observe(self.node_index, state, action, next_state, reward, done)

    def update(self, valid_steps=None):
        return self.engine.update()

    def end_episode(self):
        return self.engine.end_episode()

    def get_stats(self):
        return self.engine.get_stats()

    def save(self, path):
        self.engine.save(path)

    def load(self, path):
        self.engine.load(path)
