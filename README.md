# PointProgrammerNet：可加、定长、连续查询的点云 FWP

PointProgrammerNet 将无序点集写入固定尺寸的 Fast Weight Program（FWP）矩阵，再用任意空间坐标查询局部统计。它不依赖 FPS、KNN 或离散体素层级；点的写入可以求和，因此同一套表示既能处理静态点云，也能自然扩展到持续到来的点云帧。

当前代码首先在 ShapeNetPart 单帧部件分割上验证表示能力。流式更新、衰减和 Delta 更新已经有明确的代数形式，但尚未完成动态数据集上的系统实验，因此 README 会明确区分“已经验证的模型”和“下一阶段的流式扩展”。

## 核心思路

给定点坐标 `x_i` 和点属性 `a_i`，模型先用逐点网络生成内容特征：

```text
g_i = MLP_in([x_i, a_i])
k_i = spatial_key(x_i)
v_i = [1, x_i, g_i, x_i⊗g_i, sym(x_i x_i^T), g_i⊙g_i]
```

每个点通过一个外积写入 FWP：

```text
M = Σ_i k_i v_i^T = K^T V
```

对查询坐标 `q`，模型用同一空间键族读取：

```text
r(q) = k(q)^T M
```

`r(q)` 中保存的是未归一化的可加统计。解码器用质量列恢复局部均值、相对质心、密度、坐标—特征交叉矩、空间协方差和特征标准差，再经逐点 MLP 生成上下文特征。

这一步的关键不是把每个点逐字保存下来，而是把点集编译成一个可以连续寻址的固定状态。只要网络参数和坐标系不变，点的顺序、分块方式以及帧的到达顺序都不会改变求和结果。

## 定型架构

当前推荐模型名为 `pointprogrammer_normal_seg`；其 XYZ 版本为 `fwp_second_order_seg`。

```text
XYZ + normal
    │
    ├─ 主 FWP：128 × 170，正值连续空间键，初始半径 0.20
    ├─ 细 FWP： 64 × 170，连续归一化 RBF 分区，初始半径 0.15
    └─ 粗 FWP： 64 × 170，连续归一化 RBF 分区，初始半径 0.60
             │
             ├─ 三个状态独立写入、独立按原坐标读取
             └─ 细/粗读出拼接投影，以有界残差加入主分支
                         │
                    逐点 MLP
                         │
          局部特征 + 全局 max + 类别 one-hot
                         │
                    50 类部件预测
```

三个 FWP 都直接读取原始点集，没有串行地把一个压缩状态再次压缩。主分支负责稳定表示，两个辅助分支补充不同带宽下的统计；辅助信息通过残差加入，因此主表示不会被强制覆盖。

三个状态都保存二阶统计。以当前内部内容宽度 32 为例，每行的 170 列由以下部分构成：

```text
1 + 3(XYZ) + 32(g) + 3×32(XYZ×g) + 6(对称 XYZ 二阶矩) + 32(g²) = 170
```

法线只作为点属性参与 `g_i` 和值矩阵，不进入空间键。这样，地址仍由几何位置决定，而法线补充局部表面方向。旋转点云时必须对法线施加相同旋转；平移不改变法线。

辅助 RBF 槽位的中心只是在归一化空间中的连续可学习参数，Sobol 序列仅用于初始覆盖。它们不是采样出的代表点，也不是硬分区；每个输入点仍以连续权重写入固定矩阵。

## 为什么是固定尺寸

单个状态的形状为 `d_k × d_v`，与输入点数和历史帧数无关。当前三个状态共含 `128 + 64 + 64 = 256` 行，每行 170 个可加统计量。输入从 1,024 点增加到更多点或更多帧时，状态形状保持不变，只改变矩阵中的数值。

朴素批量写入的计算量为 `O(N d_k d_v)`，状态存储量为 `O(d_k d_v)`；对 `Q` 个坐标的读取量为 `O(Q d_k d_v)`。实现使用矩阵乘法，并支持把大量点分块累加。

“固定尺寸”指状态矩阵的形状固定，不代表其内容固定。它是一块随新观测持续更新的空间统计记忆。

## 可加性与流式点云

若点集 `A` 和 `B` 使用同一网络参数、同一世界坐标系和同一归一化方式，则：

```text
M(A ∪ B) = M(A) + M(B)
```

因此一帧中的所有点可以先打包为键矩阵和值矩阵：

```text
B_t = K_t^T V_t
```

最简单的无遗忘流式更新为：

```text
M_t = M_(t-1) + B_t
```

需要逐渐忘记旧点时，可使用指数衰减：

```text
M_t = ρ_t M_(t-1) + B_t,  0 ≤ ρ_t ≤ 1
```

如果已保存或能够重算一批点的贡献，也可执行删除：

```text
M_new = M_old - M(deleted_points)
```

进一步可以采用 Delta 规则，用新帧纠正旧状态已经预测出的内容：

```text
M_t = ρ M_(t-1) + β K_t^T (V_t - K_t M_(t-1))
```

其中 `K_t M_(t-1)` 是旧记忆对新点键的预测，括号内是创新残差。这种更新不需要给无序点人为指定序列，只把帧作为状态转移的时间单位。`β`、键归一化和谱稳定性需要在动态数据集上验证。

每帧更新还可以写成仿射变换 `(A_t, B_t)`：

```text
M_t = A_t M_(t-1) + B_t
(A₂,B₂) ∘ (A₁,B₁) = (A₂A₁, A₂B₁ + B₂)
```

该复合满足结合律。当 `A_t`、`B_t` 只由当前帧产生时，可以用并行前缀扫描一次求出所有帧的前缀状态，再并行进行分类或分割。

实际流式使用有两个必要条件：所有帧必须先变换到同一坐标系；不能逐帧各自居中和缩放，否则相加的空间地址没有共同含义。当前数据加载器的逐样本归一化适合 ShapeNetPart 单帧实验，动态场景应改用共享场景尺度或固定世界坐标变换。

## 核心创新点与主张边界

本项目的技术贡献集中在整体表示方式，而不是声称外积、核特征、二阶矩或残差连接本身是新技术：

1. 用固定尺寸的外积 FWP 状态表示无序点云，使空间统计对点和数据块严格可加。
2. 用连续坐标键写入和查询，同一状态可以在原始点或未观测坐标上产生平滑预测。
3. 用并行的主、细、粗 FWP 和二阶可加统计缓解单一状态的压缩损失，同时保留流式累加性质。
4. 将法线等附加属性放入值路径，保持空间寻址和语义内容的职责清晰。
5. 将静态表示自然提升为固定内存的帧级流式状态，并给出衰减、删除、Delta 更新及并行前缀计算形式。

当前已验证的是静态 ShapeNetPart 分割。动态流式精度、长期漂移、位姿误差和 Delta 更新稳定性仍需专门实验，论文中应作为待验证贡献，不能写成已经取得的结果。训练阶段的 BatchNorm 会使用批统计；严格的分块/流式等价性应在参数和归一化统计冻结后的推理阶段讨论。

## 环境与数据

推荐环境：

- Python 3.10+
- PyTorch（CUDA 版本）
- NumPy
- h5py

数据集使用 ShapeNetPart。法线模式读取公开 normal-point 格式中的 `x y z nx ny nz label`；只归一化 XYZ，并重新单位化法线。训练集由官方 train 与 val 组成，测试 ID 被排除并去重。不同点数的样本会被确定性地取样或重复到 `--num-points`。

首次运行法线实验会建立独立缓存。训练增强只对 XYZ 加抖动，不修改法线方向。

## 快速复现

在项目根目录运行三随机种子的五轮快速比较：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\run_normal_pointprogrammer_probe.ps1"
```

完整 20 轮比较：

```powershell
& "C:\Users\13633\miniconda3\envs\qwen_imdb\python.exe" .\train_seg.py `
  --models "pn2_normal_seg,pointprogrammer_normal_seg" `
  --seeds "0,1,2" --epochs 20 --num-points 1024 `
  --batch-size 16 --eval-batch-size 16 --use-normals `
  --save-checkpoints --output-dir ".\artifacts_seg\normal_full"
```

XYZ 定型模型：

```powershell
& "C:\Users\13633\miniconda3\envs\qwen_imdb\python.exe" .\train_seg.py `
  --models "fwp_second_order_seg" --seeds "0,1,2" `
  --epochs 20 --num-points 1024 --batch-size 16 `
  --eval-batch-size 16 --save-checkpoints
```

不加 `--force` 时，程序会复用同一实验身份下已完成的结果；加上 `--force` 才会重新训练。输出包含 `segmentation_results.json` 和可选检查点。

## 当前结果

ShapeNetPart，1,024 点，快速协议为 5 epochs、2,048 个训练样本、512 个测试样本、3 个随机种子：

| 模型 | Point accuracy | Instance mIoU | Category mIoU | 参数量 |
|---|---:|---:|---:|---:|
| PointNet++，XYZ+normal | 91.32 ± 0.11 | 79.58 ± 0.26 | 71.48 | 573,682 |
| PointProgrammerNet，XYZ+normal | 91.30 ± 0.04 | 78.90 ± 0.03 | **72.33** | **497,810** |

XYZ 版 PointProgrammerNet 的完整 20 轮三种子结果为：`93.12 ± 0.04` point accuracy、`83.36 ± 0.08` instance mIoU、`78.73` category mIoU，参数量 `497,522`。

快速结果表明：加入法线后，PointProgrammerNet 的逐点准确率已与 PointNet++ 基本持平，category mIoU 更高，参数量约少 13%；instance mIoU 仍低 0.68。五轮子集结果只用于快速比较，最终论文表格应以完整训练、多随机种子和统一评测协议为准。

## 代码入口

- `fwpnet_core/layers.py`：连续键、FWP 写入/读取、二阶统计。
- `fwpnet_core/segmentation.py`：三状态 PointProgrammerNet 与分割头。
- `fwpnet_core/data.py`：ShapeNetPart、法线读取与缓存。
- `fwpnet_core/segmentation_experiment.py`：模型注册、训练和评测。
- `train_seg.py`：命令行入口。
- `run_normal_pointprogrammer_probe.ps1`：法线快速对照实验。

旧版 README 中的完整探索记录保存在 `EXPERIMENT_HISTORY.md`，其中包含失败消融；这些实验用于确定最终设计，不代表推荐架构。
