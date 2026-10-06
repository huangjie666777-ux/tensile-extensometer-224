# tensile_dic224

无前端的二维散斑图位移/应变测量后端（FastAPI + NumPy + SciPy）。对 8 位灰度 PNG
参考图与变形图逐点进行**整数平移搜索 + 六参数局部仿射亚像素优化**，再由有效位移的
3×3 邻域平面拟合恢复 Green–Lagrange 应变。

## 安装与启动

```bash
.venv/bin/pip install -e .          # 或直接使用已就绪的 .venv
.venv/bin/python -m tensile_dic224 # 监听 127.0.0.1:8000
# 交互式文档: http://127.0.0.1:8000/docs
```

## 可复现合成示例

```bash
.venv/bin/python examples/generate_sample.py
# 生成 examples/sample/{reference.png,deformed.png,truth.json}
```

curl 测量（JSON）：

```bash
curl -s http://127.0.0.1:8000/analyze \
  -F reference=@examples/sample/reference.png \
  -F deformed=@examples/sample/deformed.png \
  -F scale_mm_per_px=0.05 -F roi_x=32 -F roi_y=32 -F roi_w=192 -F roi_h=192 \
  -F subset_size=31 -F grid_step=16 -F search_radius=8 -F max_iterations=50 | python -m json.tool
```

下载 ZIP：

```bash
curl -s -o result.zip http://127.0.0.1:8000/download \
  -F reference=@examples/sample/reference.png \
  -F deformed=@examples/sample/deformed.png \
  -F scale_mm_per_px=0.05 -F roi_x=32 -F roi_y=32 -F roi_w=192 -F roi_h=192 \
  -F subset_size=31 -F grid_step=16 -F search_radius=8 -F max_iterations=50
unzip -l result.zip
```

## 请求格式（multipart/form-data）

| 字段 | 含义 | 约束 |
| --- | --- | --- |
| `reference`, `deformed` | 8 位单通道灰度 PNG | 同尺寸；每边 ≤ 512 px |
| `scale_mm_per_px` | 毫米/像素 | > 0 |
| `roi_x`,`roi_y`,`roi_w`,`roi_h` | 矩形 ROI（像素，左上原点） | 完全落在图内，w,h > 0 |
| `subset_size` | 奇数子区边长（px） | 正奇数 |
| `grid_step` | 网格步距（px） | 正整数；端点自动补点 |
| `search_radius` | 整数搜索半径（px） | 0–16 |
| `max_iterations` | 亚像素最大迭代数 | 1–100 |

网格点至多 400 个，超出或任何图像/参数非法时整请求返回 **422**，不产生部分结果。
请求间不共享状态。

## 坐标系与输出

- 坐标原点为图像左上角，**x 向右、y 向下**；报告的是**参考构型**网格，保留全部请求点。
- 位移 `u_mm`/`v_mm` 为像素位移乘 `scale_mm_per_px`。
- 相关系数 `zncc ∈ [-1,1]`，由零均值归一化 SSD 的最优仿射亮度拟合给出
  （`ZNCC = 1 − ZNSSD/2`），容忍整体亮度偏移与正比例变化。
- 每点先做半径内穷举整数平移，再对 `[u,v,du/dx,du/dy,dv/dx,dv/dy]` 六参数
  局部仿射映射做双三次采样的阻尼 Gauss–Newton 优化；这是**逐点局部配准**，不是整图配准。
- 逐点失效原因：`flat_texture`、`out_of_bounds`、`integer_search_out_of_bounds`、
  `subpixel_out_of_bounds`、`flat_deformed_texture`、`degenerate_jacobian`、
  `solver_failure`、`not_converged`。失效点位移/应变留空，**绝不用零位移补齐**。
- 应变：在每个有效点的 **3×3 网格邻域**内分别对 u、v 做二维线性最小二乘拟合，
  有效点少于 3 个或点共线（秩退化）时该点应变无效（留空，不插补）。由梯度构造
  变形梯度 `F`，输出 `E = (FᵀF − I)/2` 的 `Exx`、`Eyy`、`Exy`；`Exy` 为张量剪切
  （= 工程剪切的一半）。位移单位毫米，应变无量纲。

## ZIP 内容

- `points.csv`：逐点 `grid_x_mm,grid_y_mm,valid,u_mm,v_mm,zncc,converged,
  iterations,exx,eyy,exy,failure_reason`，缺测为空字段。
- `valid_mask.png`：与网格行列对齐的 8 位图，有效点 255、失效点 0。
- `result.json`：回显参数、坐标系、点数/有效点数统计与失败原因计数。

## 适用范围

仅测**面内**小/中等变形的散斑图；假设子区内仿射近似成立、散斑纹理充分且两幅图像
成像条件接近（允许亮度的仿射变化）。不处理离面运动、大旋转、透视畸变、相机标定与
三维重构。边缘点要求整数搜索中子区整体位于画面内。

## 测试

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q tensile_dic224
```

## 拉伸试验虚拟引伸计（/tensile/*）

在原有网格与逐点相关之上，新增拉伸试验虚拟引伸计接口（旧接口不变）。每帧
独立对同一参考图测量，**不累积位移**。

生成可复现拉伸示例：

```bash
.venv/bin/python examples/generate_tensile_sample.py
# 生成 examples/tensile/{reference.png,frames.zip,forces.csv,truth.json}
```

curl 分析（JSON）：

```bash
curl -s http://127.0.0.1:8000/tensile/analyze \
  -F reference=@examples/tensile/reference.png \
  -F frames=@examples/tensile/frames.zip \
  -F forces=@examples/tensile/forces.csv \
  -F scale_mm_per_px=0.05 -F roi_x=32 -F roi_y=32 -F roi_w=192 -F roi_h=192 \
  -F subset_size=31 -F grid_step=16 -F search_radius=8 -F max_iterations=50 \
  -F area_mm2=25 -F p1_x=64 -F p1_y=128 -F p2_x=192 -F p2_y=128 \
  -F fit_strain_min=0.002 -F fit_strain_max=0.008 | python -m json.tool
```

下载 ZIP（把 /tensile/analyze 换成 /tensile/download 并加 -o result.zip）。

### 输入

- `reference`：8 位灰度参考 PNG（**16 位 PNG 一律拒绝**）。
- `frames`：2–12 个变形帧 PNG 的 ZIP，文件名（去扩展名）即 frame_id。
- `forces`：CSV，列 `frame_id,time_s,force_N`；frame_id 唯一且与 ZIP 内图片
  一一对应，time_s 有限且严格递增（行序即序列），force_N 为非负有限牛顿值。
- `area_mm2`：正有限原始截面积 mm²。
- `p1_x,p1_y,p2_x,p2_y`：参考图像素坐标的两个不同引伸计端点，必须落在网格
  覆盖范围内。
- `fit_strain_min,fit_strain_max`：非负且递增的拟合应变闭区间。
- 其余 DIC 参数同旧接口。任何非法输入整请求 422，不产生部分结果。

### 测量与计算

- 端点位移：仅当端点所在网格单元**四角位移全部有效**时做双线性插值，
  否则该帧标距无效（gauge_valid=0），不补缺测。
- 工程应变：变形后两端点欧氏距离 / 原始标距 − 1（统一毫米坐标与位移）。
- 工程应力：force_N / area_mm2，单位 MPa（N/mm²）。
- 拟合：闭区间内全部有效点最小二乘 σ = Eε + b，需至少 3 个不同应变且
  E > 0，返回 E、b、R²；失败时给出 reason，曲线仍完整保留。
- 屈服：拟合上界之后按 0.2% 偏移线 σ = E(ε − 0.002) + b，仅在**相邻有效
  且应变递增**的帧间寻找实测减直线由正到非正的首次交点并线性插值；不跨
  缺测、不外推，无交点返回 null 及原因。

### ZIP 内容（跨文件以 frame_id 联动）

- `curve.csv`：全部帧 frame_id,time_s,force_N,stress_MPa,eng_strain,
  gauge_valid,gauge_reason，缺测应变留空。
- `frames.csv`：逐帧端点位移、标距、应变、应力与有效点数。
- `frames/<frame_id>_points.csv`：逐帧 DIC 逐点结果（同旧接口 points.csv）。
- `tensile_result.json`：参数回显、帧序列、拟合与屈服结果、文件清单。
