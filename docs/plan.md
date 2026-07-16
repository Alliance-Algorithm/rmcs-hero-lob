# LobVisionCore 在线移植方案

## 背景

`pipeline.hpp` 中算法阶段 2-6（ORB 配准、前景提取、轨迹累积、图像合成、压缩）已从测试环境验证并移植，但测试环境是离线模式（读取视频文件 → 输出单张图片），不能直接用于实时场景。

`lob_vsison_core.cpp` 当前只有骨架，需要将其实现为在线模式：
- `update()` 以 1000Hz 运行，图像处理耗时长，需要独立 worker 线程
- 接收摄像头帧（~120fps）和发射信号，实时输出长曝光结果

---

## 整体架构

```
update() 主线程 (1000Hz)                Worker 线程 (按帧驱动)
┌──────────────────────┐        ┌────────────────────────────────────┐
│ CameraFrame 入队     │        │                                    │
│ bullet_fired 检测    │ frame  │  状态机 + Pipeline 串联            │
│                      │ queue  │                                    │
│ trigger_pending_ 标志│ ──────→│  IDLE → WAITING → PROCESSING       │
│                      │        │       → SYNTHESIZING → DONE        │
└──────────────────────┘        └────────────────────────────────────┘
                                          │
                                          ▼
                                   ┌──────────────┐
                                   │ Output 接口   │
                                   │ background   │
                                   │ track        │
                                   │ exposure     │
                                   └──────────────┘
```

## 职责划分

### `LobVisionCore` — 在线调度与控制

| 职责 | 说明 |
|------|------|
| 历史队列维护 | 每 N 帧采样一次入队，最大 15 帧（~0.5s） |
| 触发检测 | 检测 `bullet_fired_` 上升沿 |
| 状态机 | 控制 IDLE → WAITING → PROCESSING → SYNTHESIZING → DONE |
| Worker 线程 | `before_updating()` 启动，析构函数 join |
| 线程安全队列 | `std::deque<CameraFrame>` + `std::mutex` + `std::condition_variable`，FIFO 逐帧消费，处理慢时队列自然堆积，不会丢帧 |

### `Pipeline` — 算法流水线

| 方法 | 说明 |
|------|------|
| `SetReferenceFromHistory(images, frame_id)` | 对历史队列做中值法提取参考帧，设置给阶段 1 |
| `ProcessFrame(frame)` | 在线处理单帧：阶段2(ORB配准) → 阶段3(前景) → 阶段4(轨迹累积) |
| `Finalize()` | 阶段5(CLAHE+融合) → 阶段6(压缩) → 返回最终图像 |
| `ResetTracker()` | 重置阶段4轨迹累积状态（每次触发时调用） |

### `ReferenceFrameSelector` — 阶段 1

| 方法 | 说明 |
|------|------|
| `SetReference(bgr)` | 直接设置外部传入的参考帧 |
| `GetReferenceAsFrameData()` | 将参考帧包装为 `FrameData` 供后续阶段使用 |

原有 `Process(FrameData, TrackingResult)` 接口保留不动，离线 `Pipeline::Run()` 仍可使用。

---

## 状态机

触发后第 1-60 帧丢弃，第 61-300 帧每帧走完整流水线。`process_start_frame` 和 `process_end_frame` 定义参与处理的帧范围。

| 状态 | 行为 |
|------|------|
| **IDLE** | 维护历史队列（每 `history_sample_interval` 帧采样一次入队，上限 `history_queue_max_size`）。当检测到触发信号：拷贝历史队列 → `pipeline_.SetReferenceFromHistory()` → `pipeline_.ResetTracker()` → `frame_counter = 0` → 切换到 `WAITING` |
| **WAITING** | 从 `frame_queue_` 逐帧出队并递增 `frame_counter`，帧数据丢弃不处理。当 `frame_counter >= process_start_frame` → 切换到 `PROCESSING` |
| **PROCESSING** | 从 `frame_queue_` 逐帧出队：构建 `FrameData` → `pipeline_.ProcessFrame()`（阶段2+3+4）→ 更新 `latest_shooting_track_`。当 `frame_counter >= process_end_frame` → 切换到 `SYNTHESIZING` |
| **SYNTHESIZING** | `pipeline_.Finalize()` → 写入 3 个 Output。切换到 `DONE` |
| **DONE** | 静默，输出已就绪 |

> 关键：无论处理耗时多少，worker 始终从 deque 队头逐帧弹出，不会跳帧。如果处理慢于帧率，deque 自然堆积，帧不会丢失。

---

## 数据流

```
CameraFrame (frame_id, image)
  │
  ▼
update()  ──→  frame_queue_ (mutex + condition_variable)
  │
  ▼
Worker:
  ├── IDLE  ──→  history_queue_  (每4帧入队, 最大15帧)
  │    │
  │    └── 触发: copy history_queue_ → pipeline_.SetReferenceFromHistory()
  │         → pipeline_.ResetTracker() → frame_counter=0 → 切换到 WAITING
  │
  ├── WAITING (丢弃帧 1~60)
  │    │   从 deque 逐帧出队, frame_counter++, 帧丢弃
  │    └── frame_counter >= process_start_frame → 切换到 PROCESSING
  │
  ├── PROCESSING (处理帧 61~300, 每帧走完整流水线)
  │    │   从 deque 逐帧出队: pipeline_.ProcessFrame(frame)
  │    │       │
  │    │       ├── ImageRegistratorOrb::Process(ref, frame)     → 阶段2
  │    │       ├── BackgroundRemover::Process(ref, registration) → 阶段3
  │    │       └── TrackerProcessorFast::Process(foreground)     → 阶段4
  │    │
  │    ├── 更新: *latest_shooting_track_ = trajectory.trajectory_layer
  │    └── frame_counter >= process_end_frame → 切换到 SYNTHESIZING
  │
  └── SYNTHESIZING  ──→  pipeline_.Finalize()
       │                      │
       │                      ├── ImageSynthesis::Process(ref, trajectory) → 阶段5
       │                      └── Compression::Process(synthesis)           → 阶段6
       │
       └── 写入:
            *latest_shooting_background_     = 参考帧 (中值法结果)
            *latest_shooting_track_          = trajectory_layer (轨迹层)
            *latest_shooting_exposure_image_ = 最终合成图像
```

---

## 修改清单

| # | 文件 | 操作 | 内容 |
|---|------|------|------|
| 1 | `configs.hpp` | **修改** | 新增 4 个配置字段（见下方） |
| 2 | `1_reference_frame_selector.hpp` | **修改** | 新增 `SetReference(cv::Mat)` 和 `GetReferenceAsFrameData()` 方法声明 |
| 3 | `src/1_reference_frame_selector.cpp` | **新建** | 构造函数 + `SetReference` + `GetReferenceAsFrameData` + 保留原有 `Process` 声明 |
| 4 | `pipeline.hpp` | **修改** | 新增 `SetReferenceFromHistory`、`ProcessFrame`、`Finalize`、`ResetTracker` 方法 |
| 5 | `src/lob_vsison_core.cpp` | **修改** | 完整实现：frame_queue_ + 历史队列 + 状态机 + worker 线程 |

---

## 新增配置字段 (`configs.hpp`)

```cpp
struct PipelineConfig {
    // ... 现有字段保持不变 ...

    int process_start_frame = 61;        // 触发后第61帧开始处理（1-60帧丢弃）
    int process_end_frame = 300;         // 触发后第300帧停止处理
    int history_queue_max_size = 15;     // 历史队列最大长度
    int history_sample_interval = 4;     // 每 N 帧采样一次入队
};
```

---

## PlatformAdapterBase 接口

根据文档要求，最终输出 3 个图像：

| Output | 类型 | 内容 |
|--------|------|------|
| `latest_shooting_background_` | `OutputInterface<cv::Mat>` | 中值法得到的背景（参考帧） |
| `latest_shooting_track_` | `OutputInterface<cv::Mat>` | 轨迹层 |
| `latest_shooting_exposure_image_` | `OutputInterface<cv::Mat>` | 拼合后的最终图像 |

---

## 注意事项

1. **`FrameData::timestamp_seconds`**：`CameraFrame` 只有 `frame_id` 没有时间戳，通过 `frame_id / 120.0` 估算（帧率 120fps）
2. **不阻塞主线程**：`update()` 只做入队和标志位读写，所有图像处理在 worker 线程
3. **阶段 2-6 状态管理**：`ImageRegistratorOrb` 缓存了参考帧的 ORB 特征（`cached_ref_frame_index_`），触发时参考帧变化会自动重新提取；`TrackerProcessorFast` 需要 `Reset()` 清除上一轮累积
4. **中值法背景提取**：对 N 张历史图像逐像素取中位数，生成背景图作为参考帧
5. **`process_start_frame` 和 `process_end_frame` 均可通过 ROS 参数配置**
