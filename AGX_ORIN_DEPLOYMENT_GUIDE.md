# NVIDIA AGX Orin 部署指南

本指南详细说明如何在 NVIDIA AGX Orin 上部署带有 TensorRT 优化的多相机目标跟踪系统,并配置远程地面站监控。

## 目录

1. [系统要求](#系统要求)
2. [TensorRT 模型优化](#tensorrt-模型优化)
3. [系统部署](#系统部署)
4. [地面站远程监控](#地面站远程监控)
5. [性能优化](#性能优化)
6. [故障排除](#故障排除)

---

## 系统要求

### 硬件要求

- **平台**: NVIDIA AGX Orin (32GB/64GB)
- **相机**: 4 × 1080P 相机 (360°×80° 视场)
- **存储**: ≥64GB NVMe SSD
- **网络**: 千兆以太网 (用于地面站连接)
- **可选**: LiDAR 用于 3D 测距

### 软件要求

- **操作系统**: Ubuntu 20.04/22.04 (JetPack 5.x+)
- **ROS**: ROS 2 Humble/Foxy
- **CUDA**: 11.4+ (JetPack 自带)
- **TensorRT**: 8.5+ (JetPack 自带)
- **Python**: 3.8+

---

## TensorRT 模型优化

TensorRT 是 NVIDIA 的高性能深度学习推理优化器,可将推理速度提升 2-5 倍。

### 1. 安装依赖

```bash
cd /path/to/ultralytics_ros1
pip3 install -r requirements.txt
```

### 2. 单个模型转换

#### 转换用于单相机 (640×640)

```bash
python3 script/convert_to_tensorrt.py \
    --model models/yolov8m.pt \
    --imgsz 640 \
    --half \
    --batch 1 \
    --workspace 4 \
    --device 0
```

#### 转换用于四目相机 (1920×1080)

```bash
python3 script/convert_to_tensorrt.py \
    --model models/yolov8m.pt \
    --imgsz 1920 1080 \
    --half \
    --batch 1 \
    --workspace 4 \
    --device 0 \
    --benchmark
```

**参数说明**:
- `--model`: YOLO 模型路径 (.pt 文件)
- `--imgsz`: 输入图像尺寸 (height width)
- `--half`: 使用 FP16 精度 (推荐,AGX Orin 原生支持)
- `--int8`: 使用 INT8 精度 (需要校准数据,最快但精度稍降)
- `--batch`: 批处理大小 (单相机用 1,多相机可尝试 4)
- `--workspace`: TensorRT 工作空间 (GB)
- `--benchmark`: 转换后进行性能测试

### 3. 批量转换多个模型

使用批处理脚本一次性转换多个模型:

```bash
# 为所有配置转换所有模型
chmod +x script/optimize_models_for_orin.sh
./script/optimize_models_for_orin.sh --all

# 仅转换用于单相机
./script/optimize_models_for_orin.sh --single-camera

# 仅转换用于多相机
./script/optimize_models_for_orin.sh --multi-camera

# 转换特定模型
./script/optimize_models_for_orin.sh --models "yolov8m yolov8l" --multi-camera
```

### 4. 模型选择建议

| 模型 | 精度 | 速度 (FPS) | 适用场景 |
|------|------|-----------|---------|
| YOLOv8n | FP16 | ~120 | 需要极高帧率 |
| YOLOv8s | FP16 | ~80 | 平衡性能 |
| YOLOv8m | FP16 | ~50 | 推荐 (平衡精度和速度) |
| YOLOv8l | FP16 | ~30 | 需要高精度 |
| YOLOv8x | FP16 | ~20 | 最高精度 |

**注**: 以上 FPS 为单相机 1080P 输入的估计值,实际性能取决于具体的 AGX Orin 型号。

### 5. 验证 TensorRT 引擎

```bash
# 列出所有生成的引擎
ls -lh models/*.engine

# 测试引擎
python3 script/convert_to_tensorrt.py \
    --model models/yolov8m_1920x1080_fp16_batch1.engine \
    --benchmark
```

---

## 系统部署

### 1. 编译 ROS 2 包

```bash
cd /path/to/ultralytics_ros1
colcon build --symlink-install
source install/setup.bash
```

### 2. 配置相机话题

编辑 launch 文件或使用命令行参数指定相机话题:

```bash
# 方法 1: 直接使用命令行参数
ros2 launch ultralytics_ros orin_multi_camera_tracker.launch.xml \
    camera0_topic:=/my_camera0/image_raw \
    camera1_topic:=/my_camera1/image_raw \
    camera2_topic:=/my_camera2/image_raw \
    camera3_topic:=/my_camera3/image_raw \
    yolo_model:=yolov8m_1920x1080_fp16_batch1.engine

# 方法 2: 修改 launch 文件的默认值
vim launch/orin_multi_camera_tracker.launch.xml
```

### 3. 启动单相机跟踪

```bash
ros2 launch ultralytics_ros orin_single_camera_tracker.launch.xml \
    yolo_model:=yolov8m_640x640_fp16_batch1.engine \
    input_topic:=/camera/image_raw \
    device:=cuda:0
```

### 4. 启动多相机跟踪

```bash
ros2 launch ultralytics_ros orin_multi_camera_tracker.launch.xml \
    yolo_model:=yolov8m_1920x1080_fp16_batch1.engine \
    device:=cuda:0
```

### 5. 使用 3D 测距 (带 LiDAR)

```bash
# 需要确保相机已标定,camera_info 话题可用
ros2 launch ultralytics_ros multi_camera_tracker_with_cloud.launch.xml \
    yolo_model:=yolov8m_1920x1080_fp16_batch1.engine \
    lidar_topic:=/velodyne_points \
    device:=cuda:0
```

---

## 地面站远程监控

地面站提供实时的Web界面,可从远程计算机监控AGX Orin上的检测和跟踪结果。

### 1. 安装依赖

```bash
# 安装 rosbridge_server (用于ROS与Web通信)
sudo apt install ros-humble-rosbridge-server

# 安装 image_transport (用于图像压缩传输)
sudo apt install ros-humble-image-transport-plugins
```

### 2. 启动完整地面站系统

```bash
ros2 launch ultralytics_ros ground_station.launch.xml \
    yolo_model:=yolov8m_1920x1080_fp16_batch1.engine \
    orin_ip:=0.0.0.0 \
    rosbridge_port:=9090 \
    web_server_port:=8000
```

这将启动:
- 多相机跟踪系统
- ROS Bridge WebSocket 服务器 (端口 9090)
- Web 服务器 (端口 8000)
- 图像压缩转换

### 3. 访问地面站界面

#### 本地访问 (在 AGX Orin 上)

```
http://localhost:8000/ground_station.html
```

#### 远程访问 (从地面站计算机)

```
http://<AGX_ORIN_IP>:8000/ground_station.html
```

例如:
```
http://192.168.1.100:8000/ground_station.html
```

### 4. 地面站功能

#### 实时监控
- ✅ 4 个相机的实时视频流 (带检测框和ID)
- ✅ 每个相机的 FPS 显示
- ✅ 全局目标统计 (总检测数、跟踪数)
- ✅ GPU 使用率监控

#### 检测目标列表
- 实时显示所有检测到的目标
- 显示目标 ID、类别、置信度、相机来源

#### 参数调整
- 置信度阈值 (conf_thres)
- IOU 阈值 (iou_thres)
- 最大检测数 (max_det)
- 实时应用参数更改

#### 360° 全景视图
- 前方相机 (0°-90°)
- 右侧相机 (90°-180°)
- 后方相机 (180°-270°)
- 左侧相机 (270°-360°)

### 5. 网络配置

#### 在同一局域网内

确保 AGX Orin 和地面站计算机在同一网络:

```bash
# 在 AGX Orin 上检查 IP
hostname -I

# 确保端口开放
sudo ufw allow 8000
sudo ufw allow 9090
```

#### 通过 WiFi 连接

如果使用 WiFi 模块:

```bash
# 配置 AGX Orin 为 AP (接入点) 模式
sudo nmcli dev wifi hotspot ssid "ORIN_AP" password "your_password"

# 或连接到现有 WiFi
sudo nmcli dev wifi connect "SSID" password "your_password"
```

#### 通过以太网直连

AGX Orin <---> 地面站笔记本电脑

```bash
# 在 AGX Orin 上设置静态 IP
sudo nmcli con add type ethernet ifname eth0 ip4 192.168.10.1/24

# 在地面站计算机上设置静态 IP
# Linux: sudo nmcli con add type ethernet ifname eth0 ip4 192.168.10.2/24
# Windows: 网络设置 -> 以太网 -> 手动设置 IP 为 192.168.10.2

# 访问地面站
http://192.168.10.1:8000/ground_station.html
```

### 6. 性能优化建议

#### 减少网络带宽

```bash
# 使用更激进的图像压缩
ros2 run image_transport republish \
    raw compressed \
    --ros-args \
    -r in:=/yolo_image_camera0 \
    -r out/compressed:=/yolo_image_camera0/compressed \
    -p compressed/jpeg_quality:=50  # 降低质量以减少带宽
```

#### 降低传输帧率

修改 `ground_station.html` 中的订阅频率,或在发布端限制:

```bash
ros2 topic hz /yolo_image_camera0  # 检查当前帧率
ros2 topic pub --rate 10 ...        # 限制到 10 Hz
```

---

## 性能优化

### 1. AGX Orin 电源模式

```bash
# 查看当前电源模式
sudo nvpmodel -q

# 设置为最大性能模式 (MAXN)
sudo nvpmodel -m 0

# 设置风扇为最大转速
sudo jetson_clocks
```

### 2. 检查 GPU 使用率

```bash
# 实时监控
sudo tegrastats

# 或使用 nvidia-smi
watch -n 1 nvidia-smi
```

### 3. 多相机性能优化

#### 并行处理

默认配置已启用时间同步的并行处理。如果相机不同步,禁用以提高吞吐量:

```xml
<param name="use_time_sync" value="false"/>
```

#### 降低分辨率

如果性能不足,考虑降低输入分辨率:

```bash
# 在相机驱动中降采样
ros2 run image_proc resize --ros-args \
    -r in/image_raw:=/camera0/image_raw \
    -r out/image_raw:=/camera0/image_resized \
    -p scale_width:=0.5 \
    -p scale_height:=0.5
```

#### 减少检测数量

```xml
<param name="max_det" value="50"/>  <!-- 降低最大检测数 -->
<param name="conf_thres" value="0.4"/>  <!-- 提高置信度阈值 -->
```

### 4. 跟踪算法优化

根据性能需求选择跟踪器:

| 跟踪器 | 速度 | 精度 | GPU 需求 |
|--------|------|------|---------|
| ByteTrack | 最快 | 中等 | 低 |
| OCSORT | 快 | 好 | 低 |
| BoTSORT | 中等 | 很好 | 中等 |
| DeepOCSORT | 慢 | 最好 | 高 (ReID) |
| StrongSORT | 最慢 | 最好 | 高 (ReID) |

```bash
# 使用最快的跟踪器
ros2 launch ultralytics_ros orin_multi_camera_tracker.launch.xml \
    boxmot_tracker:=bytetrack

# 使用精度最高的跟踪器
ros2 launch ultralytics_ros orin_multi_camera_tracker.launch.xml \
    boxmot_tracker:=deepocsort \
    reid_model:=osnet_x1_0_msmt17.pt
```

---

## 故障排除

### 问题 1: TensorRT 引擎创建失败

**症状**: 转换脚本报错 `Failed to build engine`

**解决方案**:
```bash
# 增加 workspace
python3 script/convert_to_tensorrt.py --workspace 8 ...

# 检查 CUDA 和 TensorRT 版本
python3 -c "import torch; print(torch.cuda.is_available())"
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

### 问题 2: 帧率很低

**症状**: FPS < 10

**解决方案**:
1. 检查电源模式: `sudo nvpmodel -m 0`
2. 使用更小的模型: `yolov8n` 而不是 `yolov8l`
3. 降低输入分辨率
4. 禁用可视化: 不使用 `debug:=true`

### 问题 3: 地面站无法连接

**症状**: Web界面显示"未连接"

**解决方案**:
```bash
# 检查 rosbridge 是否运行
ros2 node list | grep rosbridge

# 检查端口是否监听
netstat -an | grep 9090

# 检查防火墙
sudo ufw status
sudo ufw allow 9090
sudo ufw allow 8000

# 修改 ground_station.html 中的 IP 地址
# const ROS_BRIDGE_URL = 'ws://<AGX_ORIN_IP>:9090';
```

### 问题 4: 图像不显示

**症状**: 地面站连接成功但无图像

**解决方案**:
```bash
# 检查图像话题
ros2 topic list | grep yolo_image
ros2 topic hz /yolo_image_camera0

# 检查图像格式
ros2 topic info /yolo_image_camera0

# 使用 compressed 图像传输 (更高效)
# 修改 HTML 订阅 compressed 话题
```

### 问题 5: 跨相机跟踪 ID 不一致

**症状**: 目标移动到另一个相机时 ID 改变

**解决方案**:
```bash
# 启用时间同步
use_time_sync:=true

# 使用 ReID 跟踪器
boxmot_tracker:=deepocsort

# 增加跟踪缓冲区
track_buffer:=50

# 降低匹配阈值
match_thresh:=0.7
```

---

## 性能基准

### 测试配置

- **平台**: AGX Orin 64GB
- **电源模式**: MAXN
- **模型**: YOLOv8m + DeepOCSORT
- **输入**: 4 × 1080P @ 30 FPS
- **精度**: FP16

### 预期性能

| 场景 | FPS (每相机) | GPU 使用 | 功耗 |
|------|-------------|---------|------|
| 单相机 640×640 | ~100 | 50% | 15W |
| 单相机 1080P | ~50 | 70% | 25W |
| 4 相机 1080P (同步) | ~30 | 95% | 40W |
| 4 相机 1080P (独立) | ~35 | 95% | 42W |

---

## 自动启动配置

### 创建系统服务

```bash
# 创建服务文件
sudo nano /etc/systemd/system/orin-tracker.service
```

```ini
[Unit]
Description=AGX Orin Multi-Camera Tracker with Ground Station
After=network.target

[Service]
Type=simple
User=<your_username>
WorkingDirectory=/home/<your_username>/ultralytics_ros1
ExecStart=/bin/bash -c "source /opt/ros/humble/setup.bash && source /home/<your_username>/ultralytics_ros1/install/setup.bash && ros2 launch ultralytics_ros ground_station.launch.xml"
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# 启用服务
sudo systemctl daemon-reload
sudo systemctl enable orin-tracker.service
sudo systemctl start orin-tracker.service

# 检查状态
sudo systemctl status orin-tracker.service
```

---

## 进一步优化

### 1. 使用 INT8 量化

更高性能但需要校准数据:

```bash
python3 script/convert_to_tensorrt.py \
    --model models/yolov8m.pt \
    --int8 \
    --calibration-data /path/to/calibration/images
```

### 2. 动态批处理

处理多相机时使用批处理:

```bash
# 转换支持批处理的引擎
python3 script/convert_to_tensorrt.py \
    --batch 4 \
    --imgsz 1920 1080
```

### 3. NVENC 硬件编码

使用硬件编码器压缩视频流:

```bash
# 安装 GStreamer 插件
sudo apt install gstreamer1.0-plugins-good gstreamer1.0-plugins-bad

# 使用 NVENC 编码
ros2 run image_transport republish raw h264 ...
```

---

## 相关资源

- [Ultralytics Documentation](https://docs.ultralytics.com)
- [BoxMOT GitHub](https://github.com/mikel-brostrom/boxmot)
- [NVIDIA AGX Orin Documentation](https://developer.nvidia.com/embedded/jetson-agx-orin)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [ROS Bridge Suite](https://github.com/RobotWebTools/rosbridge_suite)

---

## 支持与反馈

如有问题或建议,请查看:
- `BOXMOT_GUIDE.md` - BoxMOT 跟踪详细指南
- `MULTI_CAMERA_GUIDE.md` - 多相机系统详细指南
- GitHub Issues
