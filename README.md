# 8GB 显存跑 H3：三针 + 快磨 + 提示词连贯化

> 一台 8 GB 显存的消费级显卡（RTX 4060 Ti）上，把 MiniMax H3 的视频生成跑到**可用、可复现、不炸显存**。
> 本文记录的是**实测出来的数字和踩过的坑**，不是官方文档的转述。

---

## 一、这套东西能做什么（实测）

```
分辨率      896 × 672（4:3）      ← 8GB 下的满帧档
帧数        124 帧（= 5.1667 秒 @ 24fps）
步数        2 步（pi-H3 int8 加速适配器）
单镜耗时    180–195 秒
锚点        3 张关键帧（首 / 中 / 末）
显存峰值    未爆（8GB 卡跑满帧）
```

**⇒ 换 1024×768 就撑不住 124 帧** —— 那档最多 90 帧（= 3.75 秒）。
**⇒ 想让分辨率更高或帧数更多，先降帧数，再谈分辨率。**（width/height 必须是 32 的倍数，否则报 `shape ... is invalid for input of size N`）

---

## 二、先决条件

### 软件

```
ComfyUI                        （本机实测 0.43.0，0 自定义节点）
Python 3.10+（ComfyUI 的 .venv 里跑）
ffmpeg                         （抽帧、音频处理、合成）
```

### 模型文件

按 ComfyUI 的目录放。文件名必须与下面完全一致（脚本里写死了名字，你可以改脚本，但别改错）：

```
models/diffusion_models/
  minimax_h3_ref2va_pruned_int8_convrot.safetensors     ← 底模，约 19.5 GB
models/loras/  （或 diffusion_models 同目录，看你的加载器）
  pi_h3_2step_dwd_adapter_int8_convrot.safetensors      ← 2 步适配器
  pi_h3_4step_dwd_adapter_int8_convrot.safetensors      ← 4 步适配器（备用）
models/text_encoders/
  qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
models/vae/
  minimax_h3_video_vae_int8_convrot.safetensors
  minimax_h3_audio_vae_fp32.safetensors
```

**⚠️ 三个硬要求：**
1. **底模必须是 `ref2va` 版本** —— `fl2va` 结构上只有首尾两个口，喂不了第 3、第 4 张锚点。
2. **pi-H3 加载器只认 `*_int8_convrot` 后缀的量化格式。** 普通 int8（比如混血版 `*_b25-49_int8`）会报 `'NoneType' object has no attribute 'quant_config'`。
3. **别用单节点加载** —— 走 `UNETLoader` + `Load LakonLab Adapter` 两段式。

### 显存

```
8 GB   896×672 / 124 帧 / 2 步        ← 本文档的目标档
8 GB   1024×768 / 90 帧 / 2 步        ← 上限
< 8 GB 降帧数优先，别降分辨率（降分辨率画面会糊，降帧数只是短）
```

---

## 三、核心思路：两条轴，不是一条

**⇒ 大多数人把"控制力"和"速度"当成一个旋钮。其实是两个独立的轴。**

```
轴一 · 控制    2 针（首+尾）       = 放飞，模型自己走，自由但容易行为对不上剧情
              4 针（+两个中间锚点） = 打桩，钉得严，但动作被切碎 ⇒ 僵硬
              3 针（+一个中间锚点） = 本文档推荐 ← 见第四节

轴二 · 引擎    Q3 GGUF + 8 步   = 慢，约 12.5 分钟/镜
              int8_convrot + pi-H3 2 步 = 快，约 3.3 分钟/镜 ← 本文档用这个
```

**⇒ 两条轴可以自由组合。本文档是「3 针 + 2 步引擎」，一镜 3.3 分钟。**

---

## 四、三针 vs 四针：为什么 3 针反而更连贯

**⇒ 这是本文档最值钱的一条，也是反直觉的一条。**

### 现象

```
              地板绿斑%     镜内帧间漂移
4 针           0.253%        20.1
3 针           0.000%        48.7
```

**⇒ 四针的绿斑是 3 针的无穷倍（3 针一帧都没有）。**
**⇒ 3 针的帧间漂移更大 —— 但那是"动作真的走得开"，不是"乱飘"。**

### 机制

**⇒ 4 针把 124 帧切成 3 段，每段只许走 1/3。模型觉得"这段跨度不够"却必须交出变化 ⇒ 那部分能量没有地方去 ⇒ 变成色斑。**

**⇒ 3 针给 0 → 62 → 123 一次大步，能量全用在动作上 ⇒ 色斑无处产生。**

**⇒ 一句话：**四针是"想动但不许动"，三针是"允许它自己走"。****

### 三针什么时候会失效

**⇒ 不是"三针永远比四针好"，它有明确边界：**

```
❌ 动作有两个独立阶段（先推到位 → 再调角度）
   ⇒ 三针只有一个中间锚点，两个阶段会被挤在同一帧里
   ⇒ 改四针，或拆成两个镜头

❌ 动作跨度极大（从画面一头走到另一头）
   ⇒ 中间锚点画不准

❌ 需要精确停位（必须停在某个位置）
   ⇒ 三针能指定终点，但中间会漂

❌ 人脸表演/视线精度
   ⇒ 三针解决动作连贯，解决不了"五官/视线"的精度
   ⇒ 那是生图模型的美颜先验，跟针数无关
```

**⇒ 三针的主场：单一连续过程（推门、抬手、旋转、拉远、松手），尤其**刚性物体运动**和**织物/流体**。**

---

## 五、提示词：四条必写

**⇒ 三针只解决"有多少空间动"，不解决"它知道要动什么"。后者全在提示词里。**

### ① 头部动作时间轴

**⇒ 最常见的失败：只有手在动，人是桩子。**

```
错的    "Her hand opens, revealing the coins."
对的    "HER HEAD IS TURNED DOWN toward her own fist (start) —
         her eyes on her hand, chin lowered.
         By the last frame her chin is level and her gaze is forward."
```

**⇒ 每镜都要能回答：这个时刻她的头在看哪。**

### ② 反站桩条款

**⇒ 一句话治"整个身体冻结"：**

```
⚠️ THE BODY IS NEVER FROZEN. Even when she is not walking, there is always
small live motion: her head turns, her chest rises and falls with breathing,
her weight shifts from one foot to the other.
```

### ③ 运镜必须给可执行数值

**⇒ `slowly` / `subtle` / `gentle` / `slight` 这四个词，模型理解成"几乎不动"。**

```
错的    "The camera slowly pushes in."
对的    "The camera pushes in by about 20% of the frame width at a constant speed,
         from a medium close-up to a tight shot of her open palm."
```

**⇒ 用「画面宽度百分比」或「角度」描述，不要用形容词。**

### ④ 时间坐标

**⇒ 5.1667 秒的镜头，写清三个时刻各是什么状态：**

```
0.00s   握拳，钱全藏
2.58s   手指展开一半，掌心露出钱的弧形边缘
5.125s  五指舒展，五枚平躺可数
```

**⇒ 这三条正好对应三张锚点图。**时间轴和锚点图对不上 ⇒ 必炸。**

---

## 六、三张锚点图：因果必须强

**⇒ 三帧不是"三张相似的照片"，是**同一条运动轨迹上的三个点**。**

**⇒ 出图之后的五条自查（能一眼验）：**

```
□ 机位是不是同一个？  三帧的拍摄位置/距离/角度一致，只有"状态"在变
                       （机位跳 = H3 会当成三件事，乱走）
□ 变化量是否单调？    a→m 的差异和 m→c 的差异是同一方向，不是"变多又变少"
□ 中间态是否具体？    "推到一半"要写清"哪几个到位了、哪几个还在路上、在哪"
□ 首帧是否接住上一镜？ 每镜第一句写清"从上一镜的什么状态来的"
□ 有没有重叠？        三帧不能有两帧几乎一样（那样中间锚点等于没有）
```

**⇒ 一条血泪：**从旧片子里挑三帧来当新片的三针，几乎一定失败** —— 因为那三帧不在同一条轨迹上。三帧必须**按轨迹重新出**。

---

## 七、一步一步跑

### 1. 起 ComfyUI

```bash
cd /path/to/ComfyUI
.venv/bin/python main.py --disable-auto-launch      # Windows: .venv\Scripts\python.exe
# 等到 http://127.0.0.1:8188/ 返回 200（约 55 秒）
```

### 2. 准备三张锚点图

放进 `ComfyUI/input/`，或用绝对路径（脚本会自己拷）。

### 3. 写视频提示词

```
prompts/shot_template.txt       ← 复制这个，按第五节填四条
```

### 4. 跑

```bash
python examples/run_shot.py \
  --prompt prompts/shot_template.txt \
  --frames a.png m.png c.png \
  --w 896 --h 672 --len 124 --steps 2
```

**⇒ 产物落 `ComfyUI/output/video/`，文件名叫 `pih3kf4_000NN_.mp4`。**

### 5. 验证

```bash
python scripts/check_video.py output/video/pih3kf4_00001_.mp4
```

**⇒ 看三个数：地板绿斑%、镜内漂移、亮度范围。判据见第八节。**

---

## 八、判据：怎么判断这一版合格

**⇒ 不要靠眼睛说"感觉有点闪"。量。**

```
指标            干净            有问题          怎么量
地板绿斑%       < 0.20%         > 0.35%         只量画面下半（地板/墙），排除植物
镜内漂移        < 12            > 20            2fps 抽帧，剔除切点后算相邻帧差
亮度漂移        < 15            > 25            全片亮度均值极差
帧间差突变      < 8             > 10 且非切点    边界外的突然跳变 = 闪
```

**⇒ 关于"绿斑"的一个坑**：检测器要**排除植物区域**（只量地板/墙面）。我最早把植物也算进去，误报过一次；后来靠"只取画面下半 28%"解决。

---

## 九、踩过的坑（带报错原文，方便搜索）

### `'NoneType' object has no attribute 'quant_config'`

**⇒ pi-H3 的加载器不认普通 int8 量化。** 换 `*_int8_convrot` 后缀的模型；或用 `UNETLoader` + `Load LakonLab Adapter` 两段式，别用单节点。

### `shape '[1, 24, 1, 1, 21, 2, 31, 2]' is invalid for input of size 63504`

**⇒ width 不是 32 的倍数。**1008 报错，896 正常。**height 同理。**

### `CUDA error: device not ready`

**⇒ OOM 或重启后 CUDA 上下文脏了。** 等 50–60 秒再跑。
**⇒ 跟 `VRAM grow failed: N bytes` 不是一回事** —— 后者是真显存不够，得降帧数。

### ComfyUI 缓存命中：30 秒返回的是上一次的文件

**⇒ 判定成功前必须看 `LastWriteTime`，不能看"队列空了"。** 队列空不代表这次跑了。

### 音频：`atempo` 用错会把音高一起改

**⇒ 线性插值重采样会移调（`12*log2(rate)` 半音），`atempo` 不会。**
**⇒ 配音对镜长时，目标时长 = 镜长 − 0.28 秒**（不是 − 0.06）。留 0.28 秒的余量，最后一个字才不会被镜边界切掉。

### 拼接后色温跳变

**⇒ 多镜拼接前先做逐通道均值/方差对齐到首帧。** 实测能把色偏从 delta 15.7 压到 <1.0，帧间漂移降 25%。

---

## 十、可替代方案（没有本文档同款环境怎么办）

**⇒ 这套流程里"H3"只是引擎。除了引擎，其他每一格都能换：**

```
锚点图怎么来    本文档用「小画家」（即图中转的 gpt-image）
                 你没有 ⇒ 本地 ComfyUI + SD/FLUX、任何能出图的东西都行
                 只要：三帧同机位、因果单调、状态具体（第六节五条）

视频提示词      只要引擎吃自然语言就行；Kling / Veo / 可灵 都能用第五节那四条

配音           本文档用 Qwen3-TTS 本地克隆
                 换任何 TTS 都行 ⇒ 关键是第八节那条「目标 = 镜长 − 0.28」

拼接/合成      ffmpeg，无替代需求

色温对齐       逐通道 mean/std 对齐，纯 numpy 就够，不需要任何模型
```

**⇒ 一句话：**方法比模型寿命长。**引擎会换，第六到第八节那套判据不会。**

---

## 十一、许可证

MIT。随便用、随便改，出事与我无关。

**⇒ 本仓库不含任何密钥、不含任何模型文件、不含任何本机绝对路径。**
**⇒ 需要 API 的地方，代码里只留 `os.environ.get(...)`，你自己填。**
