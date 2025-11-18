# Ultra lytics ROS 1 (Noetic) 快速开始指南

本项目已完全适配 ROS 1 Noetic (Ubuntu 20.04)，支持 BoxMOT 高级跟踪、TensorRT 优化和多相机 360° 全景跟踪。

## 目录

1. [系统要求](#系统要求)
2. [安装依赖](#安装依赖)
3. [编译项目](#编译项目)
4. [基础使用](#基础使用)
5. [TensorRT 优化](#tensorrt-优化)
6. [多相机系统](#多相机系统)
7. [地面站监控](#地面站监控)

---

## 系统要求

- **操作系统**: Ubuntu 20.04 LTS
- **ROS 版本**: ROS 1 Noetic (Desktop-Full)
- **Python**: 3.8+
- **硬件**:
  - CPU: 4核+
  - RAM: 8GB+
  - GPU: NVIDIA GPU (可选,用于 TensorRT 加速)
  - 相机: USB/CSI 相机或 4 个相机用于全景

---

## 安装依赖

### 1. 安装 ROS 1 Noetic

```bash
# 添加 ROS 源
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# 安装 ROS Noetic
sudo apt update
sudo apt install ros-noetic-desktop-full

# 初始化 rosdep
sudo rosdep init
rosdep update

# 环境设置
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 2. 安装系统依赖

```bash
# ROS 相关包
sudo apt install -y \
    ros-noetic-cv-bridge \
    ros-noetic-vision-msgs \
    ros-noetic-image-transport \
    ros-noetic-image-transport-plugins \
    ros-noetic-pcl-ros \
    ros-noetic-tf2-eigen \
    ros-noetic-message-filters

# Python 依赖
sudo apt install -y \
    python3-pip \
    python3-opencv \
    python3-numpy

# 可选: ROS Bridge (用于地面站)
sudo apt install -y ros-noetic-rosbridge-server
```

### 3. 安装 Python 包

```bash
cd ~/catkin_ws/src/ultralytics_ros
pip3 install -r requirements.txt
```

**requirements.txt 内容**:
```
lap==0.4.0
onnx==1.14.0
urllib3==1.26.18
numpy==1.23.4
opencv-python==4.7.0.72
ultralytics
boxmot
rospkg
```

---

## 编译项目

### 1. 创建工作空间并克隆项目

```bash
# 创建 catkin 工作空间
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src

# 克隆项目 (或复制已有项目)
git clone <your-repo-url> ultralytics_ros
cd ultralytics_ros
```

### 2. 编译

```bash
cd ~/catkin_ws
catkin_make

# 或使用 catkin build (推荐)
sudo apt install python3-catkin-tools
catkin build

# 加载环境
source devel/setup.bash
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
```

### 3. 下载 YOLO 模型

```bash
cd ~/catkin_ws/src/ultralytics_ros
mkdir -p models

# 下载默认模型 (自动下载到 ~/.cache/ultralytics)
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
python3 -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"

# 或手动下载并放置到 models/ 目录
```

---

## 基础使用

### 1. 单相机检测和跟踪

```bash
# 使用 Ultralytics 内置跟踪器
roslaunch ultralytics_ros tracker.launch \
    yolo_model:=yolov8m.pt \
    input_topic:=/camera/image_raw \
    device:=cpu

# 使用 BoxMOT 跟踪器
roslaunch ultralytics_ros tracker.launch \
    yolo_model:=yolov8m.pt \
    tracking_method:=boxmot \
    boxmot_tracker:=deepocsort \
    device:=cpu
```

**可用的 BoxMOT 跟踪器**:
- `bytetrack` - 最快,轻量级
- `botsort` - 平衡性能和精度
- `deepocsort` - 高精度,需要 ReID 模型
- `ocsort` - 快速且准确
- `strongsort` - 最高精度,需要 ReID 模型
- `hybridsort` - 混合策略

### 2. 查看结果

**在新终端中**:
```bash
# 查看检测结果
rostopic echo /yolo_result

# 查看图像
rosrun image_view image_view image:=/yolo_image

# 查看话题列表
rostopic list | grep yolo
```

### 3. 调整参数

```bash
roslaunch ultralytics_ros tracker.launch \
    conf_thres:=0.3 \
    iou_thres:=0.5 \
    max_det:=100 \
    device:=cuda:0 \
    debug:=true
```

---

## TensorRT 优化

### 1. 转换模型为 TensorRT (AGX Orin/Jetson)

```bash
# 单相机 640×640
python3 scripts/convert_to_tensorrt.py \
    --model models/yolov8m.pt \
    --imgsz 640 \
    --half \
    --batch 1

# 四目相机 1920×1080
python3 scripts/convert_to_tensorrt.py \
    --model models/yolov8m.pt \
    --imgsz 1920 1080 \
    --half \
    --batch 1 \
    --benchmark
```

### 2. 批量转换

```bash
chmod +x scripts/optimize_models_for_orin.sh
./scripts/optimize_models_for_orin.sh --all
```

### 3. 使用 TensorRT 模型

```bash
roslaunch ultralytics_ros tracker.launch \
    yolo_model:=yolov8m_640x640_fp16_batch1.engine \
    device:=cuda:0
```

---

## 多相机系统

### 1. 四目360° 全景跟踪

```bash
roslaunch ultralytics_ros multi_camera_tracker.launch \
    yolo_model:=yolov8m.pt \
    camera0_topic:=/camera0/image_raw \
    camera1_topic:=/camera1/image_raw \
    camera2_topic:=/camera2/image_raw \
    camera3_topic:=/camera3/image_raw \
    device:=cuda:0
```

### 2. AGX Orin 优化版本

```bash
roslaunch ultralytics_ros orin_multi_camera_tracker.launch \
    yolo_model:=yolov8m_1920x1080_fp16_batch1.engine \
    device:=cuda:0
```

### 3. 查看多相机结果

```bash
# 查看各相机检测结果
rostopic echo /yolo_result_camera0
rostopic echo /yolo_result_camera1

# 查看各相机图像
rosrun image_view image_view image:=/yolo_image_camera0
rosrun image_view image_view image:=/yolo_image_camera1
```

---

## 地面站监控

### 1. 启动 ROS Bridge

```bash
# 启动 rosbridge_server
roslaunch rosbridge_server rosbridge_websocket.launch
```

### 2. 启动 Web 服务器

```bash
# 在新终端
cd ~/catkin_ws/src/ultralytics_ros
python3 scripts/web_server.py --port 8000
```

### 3. 访问地面站

在浏览器中打开:
```
http://localhost:8000/ground_station.html
```

远程访问:
```
http://<AGX_ORIN_IP>:8000/ground_station.html
```

**地面站功能**:
- ✅ 4 相机实时视频流
- ✅ 检测和跟踪可视化
- ✅ FPS 监控
- ✅ 参数实时调整
- ✅ 统计信息展示

---

## 常用命令

### 启动核心功能

```bash
# 1. 启动 roscore (第一个终端)
roscore

# 2. 启动相机驱动 (第二个终端)
# USB 相机
rosrun usb_cam usb_cam_node

# 或 CSI 相机 (Jetson)
roslaunch jetson_camera jetson_camera.launch

# 3. 启动跟踪器 (第三个终端)
roslaunch ultralytics_ros tracker.launch device:=cuda:0

# 4. 查看结果 (第四个终端)
rosrun image_view image_view image:=/yolo_image
```

### 调试命令

```bash
# 查看节点列表
rosnode list

# 查看话题列表
rostopic list

# 查看话题信息
rostopic info /yolo_result

# 查看话题频率
rostopic hz /yolo_image

# 查看节点信息
rosnode info /tracker_node

# 录制数据
rosbag record -a

# 播放数据
rosbag play your_bag.bag
```

---

## 性能优化

### 1. GPU 加速

```bash
# 使用 CUDA
roslaunch ultralytics_ros tracker.launch device:=cuda:0

# AGX Orin 最大性能模式
sudo nvpmodel -m 0
sudo jetson_clocks
```

### 2. 降低计算负载

```bash
# 使用更小的模型
roslaunch ultralytics_ros tracker.launch yolo_model:=yolov8n.pt

# 提高置信度阈值
roslaunch ultralytics_ros tracker.launch conf_thres:=0.4

# 减少最大检测数
roslaunch ultralytics_ros tracker.launch max_det:=50
```

### 3. 网络优化(地面站)

```bash
# 使用压缩图像传输
rosrun image_transport republish raw compressed \
    _input:=/yolo_image \
    _output:=/yolo_image/compressed \
    _compressed/jpeg_quality:=50
```

---

## 故障排除

### 问题 1: 找不到 ultralytics_ros 包

```bash
# 确保已编译并加载环境
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

### 问题 2: 找不到 YOLO 模型

```bash
# 检查模型路径
roscd ultralytics_ros
ls models/

# 或使用绝对路径
roslaunch ultralytics_ros tracker.launch \
    yolo_model:=/path/to/yolov8m.pt
```

### 问题 3: BoxMOT 导入错误

```bash
# 重新安装 boxmot
pip3 install --upgrade boxmot
```

### 问题 4: CV Bridge 错误

```bash
# 重新安装 cv_bridge
sudo apt install --reinstall ros-noetic-cv-bridge
```

---

## 示例应用

### 1. 车载环视系统

```bash
# 4 相机 360° 跟踪 + TensorRT 优化
roslaunch ultralytics_ros orin_multi_camera_tracker.launch \
    yolo_model:=yolov8m_1920x1080_fp16_batch1.engine \
    camera0_topic:=/front_camera/image_raw \
    camera1_topic:=/right_camera/image_raw \
    camera2_topic:=/back_camera/image_raw \
    camera3_topic:=/left_camera/image_raw \
    device:=cuda:0
```

### 2. 移动机器人

```bash
# 单相机实时跟踪
roslaunch ultralytics_ros tracker.launch \
    yolo_model:=yolov8n.pt \
    input_topic:=/camera/image_raw \
    device:=cuda:0 \
    tracking_method:=boxmot \
    boxmot_tracker:=bytetrack
```

### 3. 安防监控

```bash
# 多相机监控 + 地面站
# 终端 1: 启动跟踪
roslaunch ultralytics_ros multi_camera_tracker.launch

# 终端 2: 启动 rosbridge
roslaunch rosbridge_server rosbridge_websocket.launch

# 终端 3: 启动 web 服务器
python3 scripts/web_server.py

# 浏览器访问: http://localhost:8000/ground_station.html
```

---

## 相关文档

- `BOXMOT_GUIDE.md` - BoxMOT 跟踪器详细说明
- `MULTI_CAMERA_GUIDE.md` - 多相机系统配置
- `AGX_ORIN_DEPLOYMENT_GUIDE.md` - AGX Orin 部署指南

---

## 技术支持

如遇问题,请检查:
1. ROS 环境是否正确加载
2. 所有依赖是否已安装
3. 模型文件是否存在
4. 相机话题是否正常发布

常用诊断命令:
```bash
# 检查 ROS 环境
echo $ROS_PACKAGE_PATH

# 检查包路径
rospack find ultralytics_ros

# 测试相机
rostopic hz /camera/image_raw

# 查看错误日志
rosnode list
rosnode info /tracker_node
```

---

## 快速参考

| 命令 | 说明 |
|------|------|
| `roslaunch ultralytics_ros tracker.launch` | 启动单相机跟踪 |
| `roslaunch ultralytics_ros multi_camera_tracker.launch` | 启动多相机跟踪 |
| `roslaunch ultralytics_ros orin_multi_camera_tracker.launch` | AGX Orin 优化版 |
| `python3 scripts/convert_to_tensorrt.py` | 转换 TensorRT 模型 |
| `python3 scripts/web_server.py` | 启动地面站 |
| `rostopic echo /yolo_result` | 查看检测结果 |
| `rosrun image_view image_view image:=/yolo_image` | 查看结果图像 |

祝您使用愉快! 🚀
