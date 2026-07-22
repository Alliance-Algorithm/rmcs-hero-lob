# RMCS 图像分包传输协议

## 1. 概述

本协议定义了图像数据在 RMCS 系统中的分包格式，包含 JPEG 压缩、数据分片、以及基于 Cauchy Reed-Solomon 的前向纠错（FEC）编码规则。下游接收方可根据本文档完成数据包解析、纠错恢复与 JPEG 拼合重建。

---

## 2. 常量定义

| 常量 | 值 | 说明 |
|------|-----|------|
| `kPacketSize` | 300 | 单个数据包总字节数 |
| `kHeaderSize` | 4 | 包头字节数 |
| `kPayloadSize` | 296 | 有效载荷字节数（300 - 4） |
| `kMaxPacketCount` | 255 | 单帧图像最大包序号（uint8_t 上限） |
| `kMessageTypeImage` | 0x01 | 图像消息类型标识 |
| `kFecDataPerGroup`（默认） | 10 | 每组数据包数量（k） |
| `kFecFecPerGroup`（默认） | 3 | 每组 FEC 校验包数量（r） |
| `kStatusStart` | 0x01 | 起始包标记 |
| `kStatusEnd` | 0x02 | 结束包标记 |
| `kStatusFec` | 0x03 | FEC 校验包标记 |
| `kStatusMask` | 0x0F | Status 字段低位掩码 |
| `kKPrimeShift` | 4 | k_prime 在 Status 字段中的移位量 |

> **注意：** `kFecDataPerGroup` 与 `kFecFecPerGroup` 可由配置参数 `fec_data_per_group` 和 `fec_fec_per_group` 动态指定（uint8_t 范围），并非固定值。下游实现应从配置中读取或通过协商获得。

---

## 3. 数据包格式

每个数据包固定 **300 字节**，结构如下：

| 偏移 | 长度（字节） | 字段名 | 类型 | 说明 |
|------|-------------|--------|------|------|
| 0 | 1 | `message_type` | uint8_t | 消息类型，图像数据固定为 `0x01` |
| 1 | 1 | `status` | uint8_t | 状态/标记位（详见 §3.1） |
| 2 | 1 | `image_sequence` | uint8_t | 图像序列号，范围 0~255，每帧递增（溢出回绕） |
| 3 | 1 | `packet_sequence` | uint8_t | 包序号，单帧内从 0 开始递增，跨数据包和 FEC 包统一编号 |
| 4 | 296 | `payload` | uint8_t[296] | 有效载荷（JPEG 分片数据或 FEC 校验数据） |

### 3.1 Status 字段语义

Status 字段（字节 1）按数据包角色分为两类：

#### 3.1.1 数据包

| Status 值 | 含义 |
|-----------|------|
| `0x01` | 起始数据包（packet_sequence == 0） |
| `0x00` | 中间数据包（非首包的数据包） |

> **注意：** 即使是最后一组（partial group）的数据包，status 也不使用 `0x02`，全部中间数据包均为 `0x00`。

#### 3.1.2 FEC 校验包

FEC 包的 status 字节编码了两部分信息：

```
bit[7:4] = k_prime    （最后一组的数据包个数）
bit[3:0] = 状态标记
```

| 低 4 位值 | 含义 |
|-----------|------|
| `0x03` | 非末尾 FEC 校验包 |
| `0x02` | 末尾 FEC 校验包（整帧最后 r 个 FEC 包中的最后一个） |

完整 status 值计算公式：

```
status = (k_prime << 4) | (is_last_fec ? 0x02 : 0x03)
```

其中 `is_last_fec` 仅当该 FEC 包属于最后一组且为组的最后一个 FEC 包（即组的第 r 个 FEC 包）时为真。

---

## 4. 编码流程

### 4.1 整体流程

```
原始图像 (cv::Mat)
       │
       ▼
 JPEG 压缩 (默认 quality=95)
       │
       ▼
 按 296 字节切分为 M 个原始数据包
       │
       ▼
 按 k 个一组分组 → G 组 (尾组 k_prime = M % k, 若为 0 则 k_prime = k)
       │
       ▼
 每组 k 个数据包 → Cauthy RS 编码 → 生成 r 个 FEC 包
       │
       ▼
 按组顺序输出: [组0数据包×k, 组0FEC包×r, 组1数据包×k, 组1FEC包×r, ...]
       │
       ▼
 全局 packet_sequence 从 0 连续递增
```

### 4.2 分组逻辑

设原始数据包总数 D，k = fec_data_per_group，r = fec_fec_per_group：

```
若 D == 0: 无输出

full_groups = D / k
k_prime     = D % k

若 k_prime == 0 且 full_groups > 0:
    k_prime = k
    total_groups = full_groups
否则若 k_prime > 0:
    total_groups = full_groups + 1
否则:
    total_groups = 0
```

每个全组包含 k 个数据包，尾组（若存在）包含 k_prime 个数据包。

### 4.3 输出包计数

```
总数据包数 = D
总 FEC 包数 = total_groups × r
单帧输出总包数 = D + total_groups × r
```

若总包数超过 255（kMaxPacketCount），可能发生 packet_sequence 溢出。

---

## 5. Reed-Solomon 前向纠错

### 5.1 伽罗瓦域

使用 **GF(256)**，生成多项式：

```
P(x) = x⁸ + x⁴ + x³ + x² + 1  (0x11D)
```

加法和减法均为按位异或（XOR）。乘法使用对数-指数查表法。

### 5.2 Cauchy 编码矩阵

编码矩阵为 Cauchy 矩阵，编码过程：

```
[数据向量] × [编码矩阵] = [数据 | FEC]
```

编码矩阵前 k 行为单位矩阵（对应原始数据），后 r 行为 Cauthy 矩阵。Cauthy 矩阵元素：

```
C(i, j) = 1 / (x_i + y_j)   在 GF(256) 中

其中：x_i = k + i    (i = 0, 1, ..., r-1)
      y_j = j        (j = 0, 1, ..., k-1)
```

实际实现中，每个 FEC 校验包（共 r 个）通过对 k 个数据包的对应字节进行线性组合计算：

```
FEC_payload[idx_fec][byte] = Σ(d=0..k-1) C(idx_fec, d) × data_payload[d][byte]
```

### 5.3 解码恢复

若某组中丢失/损坏了 e 个包（e ≤ r），接收方可从该组剩余 (k+r-e) 个包中任选 k 个，使用其对应系数的逆矩阵恢复原始数据：

1. 从 (k+r) 个包的编码矩阵中取出收到的 k 个包对应的行，构造 k×k 子矩阵
2. 对该子矩阵求逆（在 GF(256) 中）
3. 将逆矩阵乘以收到的 k 个包的数据得到原始 k 个数据包

> **关键约束：** 每组的 k 个数据包和 r 个 FEC 包构成独立的纠错域。各组之间不可混合恢复。

---

## 6. 接收端解包流程

```
接收 packet_sequence = 0 的数据包
       │
       ▼
 解析 header (image_sequence, status)
       │
       ▼
 持续接收直到遇到 status 低 4 位 = 0x02 的包（末包）
       │
       ▼
 从末包的 status 高 4 位提取 k_prime
       │
       ▼
 按顺序推导分组边界:
   - 前 (total_groups-1) 组: 每组 k 数据 + r FEC
   - 尾组: k_prime 数据 + r FEC
   - total_groups = (D + total_groups × r - 包数) 反推
       │
       ▼
 对每组检查丢包，必要时用 FEC 恢复
       │
       ▼
 拼接所有数据包 payload → 完整 JPEG 字节流
       │
       ▼
 cv::imdecode 解码为原始图像
```

### 6.1 分组边界推导

已知：k = fec_data_per_group, r = fec_fec_per_group, k_prime（从末 FEC 包 status 高 4 位获取）

接收到的总包数 N = 最后一个包的 packet_sequence + 1

分组关系：
```
设 total_groups = G

G = (N - D) / r
  其中 D = (G-1) × k + k_prime
         = G × k - (k - k_prime)

代入：N = D + G × r = G × k - (k - k_prime) + G × r = G × (k + r) - (k - k_prime)

因此：G = (N + (k - k_prime)) / (k + r)
     D = N - G × r
```

### 6.2 丢包恢复

对每个组（索引 g = 0, 1, ..., G-1）：
- 组 g 的数据包数：kg = (g == G-1) ? k_prime : k
- 组 g 应有包数：kg + r
- 若收到的组内包数 < kg：尝试用组内 FEC 恢复
- 若丢失超过 r 个包：该组不可恢复

---

## 7. 配置参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `interface_name` | string | (必填) | 输入/输出接口名 |
| `input_interface_name` | string | 同 interface_name | 输入接口名（可选覆盖） |
| `message_type` | int (0~255) | 1 | 消息类型标识 |
| `fec_data_per_group` | int | 10 | 每组数据包数 (k) |
| `fec_fec_per_group` | int | 3 | 每组 FEC 包数 (r) |
| `jpeg_quality` | int | 95 | JPEG 压缩质量 (0~100) |

---

## 8. 示例

### 示例 1：小图像（1 个数据包）

- D = 1, k = 10, r = 3
- k_prime = 1, total_groups = 1
- 输出 1 + 3 = 4 个包

```
packet_sequence=0: data, status=0x01 (起始)
packet_sequence=1: FEC,  status=0x12 (k_prime=1, 非末尾)
packet_sequence=2: FEC,  status=0x13 (k_prime=1, 非末尾)
packet_sequence=3: FEC,  status=0x12 (k_prime=1, 末尾位)
```

### 示例 2：中型图像（12 个数据包）

- D = 12, k = 10, r = 3
- full_groups = 1, k_prime = 2, total_groups = 2

```
组 0: 10 数据包 + 3 FEC = 13 包 (seq 0~12)
组 1: 2 数据包 + 3 FEC = 5 包  (seq 13~17)

总计: 15 数据 + 6 FEC = 21 包

packet_sequence=0~9:   data, status=0x01/0x00
packet_sequence=10~12: FEC,  status=(10<<4)|0x03 = 0xA3 (k_prime=10, 非末尾)
packet_sequence=13~14: data, status=0x00
packet_sequence=15~16: FEC,  status=(2<<4)|0x03 = 0x23 (k_prime=2, 非末尾)
packet_sequence=17:    FEC,  status=(2<<4)|0x02 = 0x22 (k_prime=2, 末尾)
```

注意：组 0 的 k_prime 值为 10（即 k），而非最终组的 k_prime=2。

### 示例 3：完整组（10 个数据包）

- D = 10, k = 10, r = 3
- k_prime = 10, total_groups = 1

```
总计: 10 数据 + 3 FEC = 13 包
末 FEC 包 status = (10<<4)|0x02 = 0xA2
```

---

## 9. 附录：GF(256) 对数和指数表

生成多项式：`0x11D`（x⁸ + x⁴ + x³ + x² + 1）

```c++
// 初始化伪代码
uint16_t value = 1;
for (uint16_t i = 0; i < 255; ++i) {
    exp_table[i] = (uint8_t)value;
    log_table[value] = (uint8_t)i;
    value <<= 1;
    if (value >= 256)
        value ^= 0x11D;
}
// 指数表扩展到 512 项以支持乘法表的快速查找
for (uint16_t i = 255; i < 512; ++i)
    exp_table[i] = exp_table[i - 255];
```

---

## 10. 修订记录

| 版本 | 日期 | 说明 |
|------|------|------|
| 1.0 | 2026-07-21 | 初始版本 |
