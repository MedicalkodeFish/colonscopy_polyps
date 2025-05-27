# --------------------------------------------------------
# Based on yolov10, modified for real-time screen capture
# https://github.com/THU-MIG/yolov10/app.py
# --------------------------------------------------------'

import gradio as gr
import cv2
import numpy as np
import mss
import time
from ultralytics import YOLO
import threading

# Global variables
streaming_active = False
current_frame = None
frame_lock = threading.Lock()
capture_thread = None # 添加一个变量来持有线程对象
show_overlay = False  # 控制是否在屏幕上显示检测框
overlay_window = None  # 用于存储覆盖窗口的引用

def capture_screen_thread(model_id, image_size, conf_threshold, update_interval):
    """后台线程：持续捕获屏幕并更新当前帧"""
    global streaming_active, current_frame, overlay_window
    
    print(f"加载模型: {model_id}")
    target_device = 'cuda' # 目标设备设为 CUDA
    actual_device = 'cpu' # 默认实际使用的设备为 CPU

    try:
        # 加载模型
        model = YOLO(model_id)
        
        # 如果需要，您可以在这里设置自定义类别名称
        # 例如：
        custom_classes = {0: "polyps"} # 替换为您的实际类别
        model.model.names = custom_classes
        
        print(f"模型初始加载设备: {model.device}") # 打印 ultralytics 自动选择的设备

        # 尝试将模型移动到目标 CUDA 设备
        try:
            # 检查是否有可用的 CUDA 设备
            import torch
            if torch.cuda.is_available():
                model.to(target_device)
                actual_device = target_device # 如果成功移动，更新实际设备
                print(f"成功将模型移动到 {actual_device} 设备。")
                # 再次检查模型内部记录的设备，虽然 predict 的 device 参数更关键
                print(f"模型内部记录设备: {model.device}")
            else:
                print(f"警告：未检测到可用的 CUDA 设备。将使用 CPU 进行推理。")
                print("请确认：")
                print("1. 已安装支持 CUDA 的 PyTorch (访问 pytorch.org 获取安装命令)。")
                print("2. NVIDIA 驱动程序已正确安装并为最新。")
                print("3. CUDA Toolkit 版本与 PyTorch 兼容。")
                actual_device = 'cpu' # 确认使用 CPU

        except Exception as e:
            print(f"尝试将模型移动到 {target_device} 时出错: {e}")
            print(f"将回退到在 {model.device} 上运行推理。请检查 CUDA 环境配置。")
            actual_device = model.device # 使用模型加载时的设备

    except Exception as e:
        print(f"加载模型失败: {e}")
        streaming_active = False
        return

    monitors = mss.mss().monitors
    monitor_to_capture = None
    if len(monitors) > 1:
        monitor_to_capture = monitors[-1]
        print(f"找到 {len(monitors)-1} 个屏幕，将从最后一个屏幕捕获: {monitor_to_capture}")
    else:
        try:
            monitor_to_capture = mss.mss().monitors[1]
            print(f"只有一个屏幕信息，尝试从主屏捕获: {monitor_to_capture}")
        except IndexError:
            print("无法确定主屏幕，请检查屏幕配置。")
            streaming_active = False
            return

    # 创建覆盖窗口（如果启用了显示覆盖）
    if show_overlay:
        cv2.namedWindow("Detection Overlay", cv2.WINDOW_NORMAL)
        if monitor_to_capture:
            # 设置窗口位置和大小与被捕获的屏幕匹配
            cv2.resizeWindow("Detection Overlay", 
                            monitor_to_capture["width"], 
                            monitor_to_capture["height"])
            # 将窗口移动到被捕获屏幕的位置
            cv2.moveWindow("Detection Overlay", 
                        monitor_to_capture["left"], 
                        monitor_to_capture["top"])
        # 设置窗口为透明和置顶
        cv2.setWindowProperty("Detection Overlay", cv2.WND_PROP_TOPMOST, 1)
        # 在Windows上设置窗口为点击穿透
        if hasattr(cv2, 'WINDOW_GUI_NORMAL'):
            cv2.setWindowProperty("Detection Overlay", cv2.WINDOW_GUI_NORMAL, cv2.WINDOW_AUTOSIZE)
        overlay_window = "Detection Overlay"

    with mss.mss() as sct:
        while streaming_active:
            try:
                # 捕获屏幕
                img = sct.grab(monitor_to_capture)
                img_np = np.array(img)  # BGRA format

                # 转换 BGRA 到 RGB (YOLO 模型需要 RGB)
                frame_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGRA2RGB)

                # 执行推理
                results = model.predict(source=frame_rgb, imgsz=image_size, conf=conf_threshold, verbose=False, device=actual_device)

                # 绘制结果
                annotated_frame = results[0].plot()
                
                # 手动交换红蓝通道 (R和B通道交换)
                annotated_frame = annotated_frame[:, :, ::-1]

                # 更新当前帧
                with frame_lock:
                    current_frame = annotated_frame

                # 如果启用了覆盖显示，则显示检测结果
                if show_overlay and overlay_window:
                    # 创建完全透明的覆盖层
                    overlay = np.zeros((monitor_to_capture["height"], monitor_to_capture["width"], 4), dtype=np.uint8)
                    # 设置完全透明的背景
                    overlay[:, :, 3] = 0
                    
                    # 从结果中获取检测框
                    boxes = results[0].boxes
                    for box in boxes:
                        # 获取坐标并调整到屏幕尺寸
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # 获取类别和置信度
                        cls = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        label = f"{results[0].names[cls]} {conf:.2f}"
                        
                        # 在覆盖层上绘制框和标签 (使用完全不透明的颜色)
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0, 255), 2)
                        cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0, 255), 2)
                    
                    # 显示覆盖层
                    cv2.imshow(overlay_window, overlay)
                    cv2.waitKey(1)  # 必须有这一行才能正确显示窗口

                # 短暂休眠以控制帧率，避免CPU占用过高
                time.sleep(update_interval) # 使用传入的更新间隔

            except mss.ScreenShotError as ex:
                print(f"屏幕捕获错误: {ex}")
                time.sleep(1) # 捕获出错时等待更长时间
            except Exception as e:
                # 捕获更广泛的异常，包括可能的 CUDA 错误
                print(f"处理帧时发生错误: {e}")
                # 检查错误是否与 CUDA 相关
                error_str = str(e).lower()
                if 'cuda' in error_str or 'nvrtc' in error_str or 'nvidia' in error_str:
                     print("错误提示与 CUDA 相关。请执行以下检查：")
                     print("1. 确认已安装支持 CUDA 的 PyTorch 版本。访问 https://pytorch.org/get-started/locally/ 并选择适合您 CUDA 版本的命令。")
                     print("2. 确认 NVIDIA 驱动程序已更新到最新版本。")
                     print("3. 确认 CUDA Toolkit 已正确安装且版本与 PyTorch 兼容。")
                     print("4. 检查 GPU 显存是否足够，或被其他程序大量占用。尝试减小图像大小 (image_size)。")
                streaming_active = False
                break # 退出循环

            # 检查是否应该停止 (在循环开始和处理后都检查)
            if not streaming_active:
                break
                
    # 关闭覆盖窗口
    if show_overlay and overlay_window:
        cv2.destroyWindow(overlay_window)
        
    print("后台捕获线程已停止。")


def get_current_frame():
    """获取当前帧用于Gradio界面更新"""
    global current_frame
    with frame_lock:
        if current_frame is None:
            # 返回一个标准尺寸的黑色图像
            return np.zeros((640, 640, 3), dtype=np.uint8)
        # 在返回给Gradio之前交换红蓝通道
        return cv2.cvtColor(current_frame.copy(), cv2.COLOR_BGR2RGB)


def start_stream(model_id, image_size, conf_threshold, update_interval, enable_overlay):
    """启动实时检测流并持续更新画面 (生成器函数)。"""
    global streaming_active, current_frame, capture_thread, show_overlay
    if streaming_active:
        print("检测已在运行中。")
        # 如果已经在运行，返回当前的帧即可
        # 使用 yield* 来正确处理生成器
        yield from (get_current_frame() for _ in range(1)) # 返回当前帧
        return

    streaming_active = True
    show_overlay = enable_overlay  # 设置是否显示覆盖层
    print(f"开始实时检测...{'启用屏幕覆盖' if show_overlay else '不启用屏幕覆盖'}")

    # 清空上一帧内容，使用传入的 image_size
    with frame_lock:
        # 确保初始空白帧尺寸与滑块设置一致
        current_frame = np.zeros((int(image_size), int(image_size), 3), dtype=np.uint8)

    # 启动后台线程
    capture_thread = threading.Thread(
        target=capture_screen_thread,
        args=(model_id, image_size, conf_threshold, update_interval), # 传递 update_interval
        daemon=True
    )
    capture_thread.start()

    # 持续产生帧给 Gradio，直到 streaming_active 变为 False
    while streaming_active:
        frame_to_yield = get_current_frame()
        yield frame_to_yield
        # 控制 Gradio 前端更新频率
        time.sleep(update_interval) # 使用传入的更新间隔

    print("实时检测流已停止。")
    # 流停止后，返回一个与当前设置匹配的空白帧以清空界面
    yield np.zeros((int(image_size), int(image_size), 3), dtype=np.uint8)


def stop_stream():
    """停止实时检测流。"""
    global streaming_active, capture_thread, overlay_window
    if streaming_active:
        print("正在停止实时检测...")
        streaming_active = False
        # 等待后台线程结束
        if capture_thread is not None:
            capture_thread.join(timeout=2.0) # 稍微增加等待时间
            if capture_thread.is_alive():
                print("警告：后台线程在超时后仍未停止。")
        capture_thread = None # 清理线程对象
        
        # 确保关闭任何可能存在的覆盖窗口
        if overlay_window:
            try:
                cv2.destroyWindow(overlay_window)
            except:
                pass
            overlay_window = None
            
        print("检测已停止。")
    else:
        print("检测尚未开始。")

    # 返回标准尺寸的空白图像，或者可以尝试获取当前的 image_size
    # 但 stop 不接收参数，所以用固定值或默认值
    return np.zeros((640, 640, 3), dtype=np.uint8)


def app():
    # 使用 gr.Blocks() 创建界面
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("选择模型和参数，然后点击 '开始实时检测'。")
                model_id = gr.Dropdown(
                    label="模型",
                    choices=[
                        "best_endo_XJ.pt",  # 替换为您实际的模型文件名
                        # 例如 "my_custom_model.pt", "my_model_v2.pt" 等
                    ],
                    value="best_endo_XJ.pt",  # 设置默认值为您的模型
                )
                image_size = gr.Slider(
                    label="图像大小 (影响性能和精度)", minimum=320, maximum=1280, step=32, value=640,
                )
                conf_threshold = gr.Slider(
                    label="置信度阈值", minimum=0.0, maximum=1.0, step=0.05, value=0.25,
                )
                update_interval = gr.Slider( # 新增滑块
                    label="更新间隔 (秒)",
                    info="每次屏幕捕获和界面更新之间的等待时间。值越小，频率越高，CPU/GPU 占用可能越高。",
                    minimum=0.01, # 约等于 100 FPS
                    maximum=1.0,  # 1 FPS
                    step=0.01,
                    value=0.05, # 默认 20 FPS
                )
                enable_overlay = gr.Checkbox(
                    label="在屏幕上显示检测框",
                    info="启用后将在被检测的屏幕上直接显示检测框",
                    value=False
                )
                with gr.Row():
                    start_button = gr.Button("开始实时检测", variant="primary")
                    stop_button = gr.Button("停止实时检测")
                    refresh_button = gr.Button("刷新画面")

            with gr.Column(scale=3):
                output_stream = gr.Image(label="实时检测画面", type="numpy", interactive=False, height=640)

        # 连接按钮事件
        start_button.click(
            fn=start_stream,
            inputs=[model_id, image_size, conf_threshold, update_interval, enable_overlay], # 添加 enable_overlay 到 inputs
            outputs=[output_stream],
        )
        
        stop_button.click(
            fn=stop_stream,
            inputs=[],
            outputs=[output_stream],
        )
        
        refresh_button.click(
            fn=get_current_frame,
            inputs=[],
            outputs=[output_stream],
        )

    return demo

# 主程序入口
if __name__ == '__main__':
    # 构建 Gradio 应用界面
    gradio_app = gr.Blocks(title="自定义模型实时屏幕目标检测")
    with gradio_app:
        gr.HTML(
            """
        <h1 style='text-align: center'>
        自定义模型实时目标检测
        </h1>
        <p style='text-align: center'>使用自定义模型捕获屏幕内容并进行实时目标检测。点击"开始实时检测"启动，点击"停止实时检测"结束。</p>
        """)
        with gr.Row():
            with gr.Column():
                app()

    print("正在启动 Gradio 应用...")
    gradio_app.launch()