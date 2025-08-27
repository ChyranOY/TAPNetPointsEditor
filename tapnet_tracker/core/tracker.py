#!/usr/bin/env python3
"""
Core tracker class for TAPNet Tracker.

Contains the main TAPNextTracker class that handles model loading, 
video processing, and track generation.
"""

import os
import jax
import tempfile
import shutil
from typing import Tuple, List, Optional, Dict
from PIL import Image
import gradio as gr
import numpy as np

from .model import forward, npload, recover_tree, create_tapnext_model
from .config import config
from ..utils.video_utils import extract_first_frame, preprocess_video
from ..utils.track_utils import generate_query_points, scale_tracks_to_original_size, save_tracks_as_pth
from ..utils.visualization import visualize_tracks, save_visualization_video
from ..utils.file_utils import scan_video_folder


class TAPNextTracker:
    """Main trajectory tracker class"""
    
    def __init__(self):
        self.model_params = None
        self.model_loaded = False
        self.video_files = {}  # Store video file information
        self.current_folder = None  # Current folder path
        # New: Track last generated file information
        self.last_video_path = None  # Last processed video path
        self.last_tracks_path = None  # Last generated trajectory file path
        self.last_original_size = None  # Last video original size (width, height)
        
        # Edit mode related attributes
        self.edit_mode_active = False
        self.edit_tracks_data = None
        self.edit_video_frames = None
        self.edit_current_frame_index = 0
        self.edit_selected_point_id = None
        self.edit_original_size = None
        self.edit_target_frame_size = None
    
    def load_model(self, checkpoint_path: str = None) -> Tuple[str, bool]:
        """Load TAPNext model"""
        try:
            if checkpoint_path is None:
                checkpoint_path = config.get_checkpoint_path()
            
            if not os.path.exists(checkpoint_path):
                return f"❌ Model file does not exist: {checkpoint_path}", False
            
            print(f"Loading model: {checkpoint_path}")
            
            # Load model parameters
            ckpt = npload(checkpoint_path)
            
            # Recover parameter tree structure
            self.model_params = recover_tree(ckpt)
            self.model_loaded = True
            
            msg = f"✅ Model loaded successfully!\n📁 Model path: `{checkpoint_path}`"
            print("Model loading completed")
            
            return msg, True
            
        except Exception as e:
            error_msg = f"❌ Model loading failed: {str(e)}"
            print(error_msg)
            self.model_loaded = False
            return error_msg, False
    
    def scan_folder(self, folder_path: str) -> Tuple[bool, str, Dict]:
        """Scan video files in folder"""
        if not folder_path or not os.path.exists(folder_path):
            return False, "❌ Please provide a valid folder path", {}
        
        try:
            # Scan video files
            self.video_files = scan_video_folder(folder_path)
            self.current_folder = folder_path
            
            if not self.video_files:
                return False, f"📁 No video files found in folder `{folder_path}`", {}
            
            from ..utils.file_utils import format_video_list_display
            display_text = format_video_list_display(self.video_files)
            
            return True, display_text, self.video_files
            
        except Exception as e:
            error_msg = f"❌ Folder scanning failed: {str(e)}"
            return False, error_msg, {}
    
    def get_video_path_by_choice(self, choice: str) -> Optional[str]:
        """Get video path based on selection"""
        if not choice or not self.video_files:
            return None
        
        # 从选择项中提取文件名（去掉状态图标）
        filename = choice.replace("✅ ", "").replace("⏳ ", "").replace("🔄 ", "").replace("❌ ", "")
        
        if filename in self.video_files:
            return self.video_files[filename]['path']
        
        return None
    
    def get_last_generated_files(self) -> Tuple[Optional[str], Optional[str]]:
        """
        获取上次生成的文件路径
        
        Returns:
            Tuple[video_path, tracks_path]: 视频和轨迹文件路径
        """
        return self.last_video_path, self.last_tracks_path
    
    def update_video_status(self, filename: str, status: str, result_path: str = None, viz_path: str = None):
        """更新视频处理状态"""
        if filename in self.video_files:
            self.video_files[filename]['status'] = status
            if result_path:
                self.video_files[filename]['result_path'] = result_path
            if viz_path:
                self.video_files[filename]['viz_path'] = viz_path
    
    def track_video(self, video_path: str, num_points: int = 32, point_method: str = "manual", 
                   output_dir: str = "outputs/", enable_visualization: bool = True, 
                   manual_points: Optional[List[Tuple[float, float]]] = None, 
                   filename: str = None) -> Tuple[str, str, str, str]:
        """
        处理视频生成轨迹
        
        Returns:
            Tuple[result_path, viz_path, message, preview_path]
        """
        if not self.model_loaded:
            return "", "", "❌ 请先加载模型", ""
        
        if not video_path or not os.path.exists(video_path):
            return "", "", "❌ 请提供有效的视频文件", ""
        
        try:
            # 更新状态为处理中
            if filename:
                self.update_video_status(filename, 'processing')
            
            # 🗂️ 新增：基于视频名创建子目录
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_output_dir = os.path.join(output_dir, video_name)
            os.makedirs(video_output_dir, exist_ok=True)
            
            print(f"📁 输出目录结构:")
            print(f"  - 基础目录: {output_dir}")
            print(f"  - 视频专用目录: {video_output_dir}")
            print(f"  - 视频名称: {video_name}")
            
            # 更新output_dir为视频专用目录
            output_dir = video_output_dir
            
            # 预处理视频
            print("📹 正在预处理视频...")
            frames_processed, frames_original, original_size = preprocess_video(
                video_path, target_size=config.TARGET_SIZE
            )
            
            # 生成查询点
            print("📍 正在生成查询点...")
            print(f"  - 点生成方法: {point_method}")
            print(f"  - 目标点数量: {num_points}")
            
            if point_method == "manual" and manual_points:
                # 手动点已经是256坐标系，直接使用
                print(f"  - 手动点数量: {len(manual_points)}")
                print(f"  - 手动点坐标: {manual_points}")
                # Manual模式：使用实际的手动点数量，而不是num_points
                query_points = generate_query_points(len(manual_points), method="manual", manual_points=manual_points)
                print(f"  - 查询点形状: {query_points.shape}")
                print(f"  - 最终查询点数量: {query_points.shape[1]} (纯手动点)")
            else:
                # 使用指定的方法生成点
                print(f"  - 使用{point_method}方法生成点")
                query_points = generate_query_points(num_points, method=point_method)
            
            # 处理查询点格式 - 按原版实现（不归一化，保持256坐标系）
            query_points_model = query_points  # 直接使用256坐标系，不归一化！
            
            print(f"🔮 开始推理...")
            print(f"  - 查询点形状: {query_points_model.shape}")
            print(f"  - 视频形状: {frames_processed.shape}")
            
            # 模型推理
            tracks_result = []
            visibles_result = []
            state = None
            
            for i, frame in enumerate(frames_processed):
                frame_batch = frame[None]  # 添加batch维度
                
                tracks, visibles, state = forward(
                    self.model_params, 
                    frame_batch, 
                    query_points_model, 
                    i, 
                    state
                )
                
                tracks_result.append(tracks)
                visibles_result.append(visibles)
                
                if (i + 1) % 10 == 0:
                    print(f"  - 已处理帧: {i + 1}/{len(frames_processed)}")
            
            # 合并结果 - 按原版实现添加坐标翻转
            tracks = np.stack(tracks_result, axis=2)[..., ::-1]  # [batch, points, frames, 2] 翻转xy
            visibles = np.stack(visibles_result, axis=2)  # [batch, points, frames, 1]
            
            print(f"✅ 推理完成")
            print(f"  - 轨迹形状: {tracks.shape}")
            print(f"  - 可见性形状: {visibles.shape}")
            
            # 🔍 调试：检查模型输出的坐标范围
            print(f"🔍 模型输出坐标范围调试:")
            print(f"  - X坐标范围: [{tracks[0, :, :, 0].min():.2f}, {tracks[0, :, :, 0].max():.2f}]")
            print(f"  - Y坐标范围: [{tracks[0, :, :, 1].min():.2f}, {tracks[0, :, :, 1].max():.2f}]")
            print(f"  - 期望范围: [0, 256] (模型输出应该在这个范围内)")
            
            # 将轨迹坐标缩放到原始视频尺寸
            tracks_scaled = scale_tracks_to_original_size(
                tracks[0], config.TARGET_SIZE, original_size
            )
            
            # 保存轨迹文件（video_name已在上面定义）
            output_filename = f"{video_name}.pth"
            saved_path = os.path.join(output_dir, output_filename)
            
            # ✅ 按原版实现：保存256x256坐标，不缩放到原始尺寸
            save_tracks_as_pth(tracks, visibles, saved_path, quant_multi=config.QUANT_MULTI)
            
            # 📁 新增：拷贝原视频到保存目录
            import shutil
            video_extension = os.path.splitext(os.path.basename(video_path))[1]
            copied_video_filename = f"{video_name}{video_extension}"
            copied_video_path = os.path.join(output_dir, copied_video_filename)
            
            try:
                print(f"📁 正在拷贝原视频到保存目录...")
                print(f"  - 原始路径: {video_path}")
                print(f"  - 目标路径: {copied_video_path}")
                shutil.copy2(video_path, copied_video_path)
                print(f"✅ 视频拷贝成功: {copied_video_filename}")
            except Exception as e:
                print(f"⚠️ 视频拷贝失败: {str(e)}")
                # 拷贝失败时使用原路径
                copied_video_path = video_path
            
            # 生成可视化（如果启用）
            visualization_path = ""
            
            if enable_visualization:
                print("🎨 正在生成可视化...")
                
                # 准备可视化数据
                tracks_for_viz = tracks_scaled[..., None, :]  # [points, frames, 1, 2]
                visibles_for_viz = visibles[0, :, :, None]  # [points, frames, 1]
                
                # 生成可视化视频
                viz_frames = visualize_tracks(
                    frames_original, tracks_for_viz, visibles_for_viz, 
                    tracks_leave_trace=config.TRACKS_LEAVE_TRACE
                )
                
                # 保存可视化视频
                viz_filename = f"{video_name}_visualization.mp4"
                visualization_path = os.path.join(output_dir, viz_filename)
                save_visualization_video(viz_frames, visualization_path, fps=config.DEFAULT_FPS)
                
                print(f"🎬 可视化视频已生成: {visualization_path}")
            
            # 更新状态为完成
            if filename:
                self.update_video_status(filename, 'completed', saved_path, visualization_path)
            
            # 🔥 新增：保存上次生成的文件信息，用于编辑模式自动填充
            self.last_video_path = copied_video_path  # 使用拷贝的视频路径
            self.last_tracks_path = saved_path
            self.last_original_size = original_size  # (width, height)
            print(f"📝 已保存文件信息用于编辑模式:")
            print(f"  - 视频: {self.last_video_path}")
            print(f"  - 轨迹: {self.last_tracks_path}")
            print(f"  - 原始尺寸: {self.last_original_size}")
            print(f"  - 专用文件夹: {output_dir}")
            print(f"  - 文件夹结构: outputs/{video_name}/")
            
            success_msg = f"""✅ 轨迹生成成功!

📁 **轨迹文件**: `{output_filename}`
📍 **完整路径**: `{saved_path}`
📊 **数据维度**: [{tracks.shape[1]}, {tracks.shape[2]}, {tracks.shape[0]}, 3]
  - {tracks.shape[1]}: 轨迹点数
  - {tracks.shape[2]}: 视频帧数  
  - {tracks.shape[0]}: batch数 (通常为1)
  - 3: [x, y, visibility]

🎯 **原始视频尺寸**: {original_size[0]} x {original_size[1]} (width x height)
📐 **推理尺寸**: {config.TARGET_SIZE[0]} x {config.TARGET_SIZE[1]} (width x height)
📁 **视频已拷贝**: `{copied_video_filename}`
🗂️ **专用文件夹**: `outputs/{video_name}/`

💡 **可在"交互式编辑"中点击"自动填充"直接使用生成的文件**"""

            if enable_visualization:
                success_msg += f"""

🎬 **可视化视频**: `{viz_filename}`"""

            return saved_path, visualization_path, success_msg, visualization_path
            
        except Exception as e:
            error_msg = f"❌ 轨迹生成失败: {str(e)}"
            print(error_msg)
            if filename:
                self.update_video_status(filename, 'failed')
            return "", "", error_msg, "" 