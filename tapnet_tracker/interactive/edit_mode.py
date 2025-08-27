#!/usr/bin/env python3
"""
Interactive editing mode for TAPNet Tracker.

Contains functions for interactive track editing, point manipulation, and coordinate updates.
"""

import os
import cv2
import torch
import tempfile
import numpy as np
from typing import Tuple, List, Dict, Optional, Any
from PIL import Image

from ..utils.visualization import generate_point_colors
from ..utils.track_utils import unzip_to_array
# 统一使用config中的输出目录配置


class TrackEditManager:
    """轨迹编辑管理器类"""
    
    def __init__(self):
        self.tracks_data = None
        self.video_path = None
        self.estimated_original_size = None
        self.original_raw_data = None
        self.original_tracks_shape = None
        self.frame_count = 0
        self.selected_point = None
        self.current_frame_index = 0
        self.point_info = []
        self.point_colors = None  # 🎨 固定的点颜色映射，确保编辑界面和可视化一致
        
    def initialize_edit_mode(self, video_path: str, tracks_path: str) -> Tuple[bool, str, Optional[Tuple]]:
        """
        进入编辑模式，加载视频和轨迹文件
        
        Args:
            video_path: 视频文件路径
            tracks_path: 轨迹文件路径
            
        Returns:
            Tuple[success, message, initial_frame_data]
        """
        try:
            # 验证文件存在
            if not os.path.exists(video_path):
                return False, f"❌ 视频文件不存在: {video_path}", None
            
            if not os.path.exists(tracks_path):
                return False, f"❌ 轨迹文件不存在: {tracks_path}", None
            
            # 获取视频信息
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False, f"❌ 无法打开视频文件: {video_path}", None
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            print(f"🎬 视频信息: {width}x{height}, {frame_count} 帧")
            
            # 使用当前视频尺寸作为原始尺寸
            original_size = (width, height)
            print(f"📏 轨迹编辑模式：使用当前视频尺寸作为原始尺寸: {original_size}")
            
            # 加载轨迹数据
            print("📊 加载轨迹文件...")
            if isinstance(tracks_path, str):
                tracks_loaded = torch.load(tracks_path)
            else:
                tracks_loaded = tracks_path
            
            tracks_np = unzip_to_array(tracks_loaded)
            
            # 记录原始数据信息
            self.original_tracks_shape = tracks_np.shape
            self.original_raw_data = tracks_np.copy()
            
            print(f"🔍 原始轨迹数据形状: {tracks_np.shape}")
            print(f"🔍 原始轨迹数据范围: min={tracks_np.min():.2f}, max={tracks_np.max():.2f}")
            
            # 处理轨迹数据用于编辑
            tracks_processed = self.process_tracks_for_editing(
                tracks_np, (width, height), original_size, quant_multi=8
            )
            
            # 设置实例变量
            self.tracks_data = tracks_processed
            self.video_path = video_path
            self.estimated_original_size = original_size
            self.frame_count = frame_count
            self.selected_point = None
            self.current_frame_index = 0
            self.point_info = []
            
            print(f"✅ 编辑模式初始化成功!")
            print(f"  - 视频: {os.path.basename(video_path)}")
            print(f"  - 轨迹: {os.path.basename(tracks_path)}")
            print(f"  - 轨迹数据形状: {tracks_processed.shape}")
            print(f"  - 视频尺寸: {original_size}")
            
            # 生成第一帧
            frame_pil, error, point_info = self.extract_frame_with_tracks(
                video_path, tracks_processed, 0, original_size
            )
            
            if frame_pil is not None:
                self.point_info = point_info
                
                return True, "✅ 编辑模式初始化成功", (frame_pil, None, [])
            else:
                return False, f"❌ 无法生成第一帧: {error}", None
                
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            return False, f"❌ 初始化编辑模式失败: {str(e)}\n详细错误:\n{error_detail}", None
    
    def process_tracks_for_editing(self, tracks_np: np.ndarray, target_frame_size: Tuple[int, int], 
                                 original_size: Tuple[int, int], quant_multi: int = 8):
        """
        处理轨迹数据以用于编辑模式
        
        Args:
            tracks_np: 原始轨迹数据
            target_frame_size: 目标帧尺寸
            original_size: 原始尺寸
            quant_multi: 量化倍数
            
        Returns:
            处理后的轨迹数据，适合编辑使用
        """
        print(f"🔧 处理轨迹数据用于编辑模式:")
        print(f"  - 输入数据形状: {tracks_np.shape}")
        print(f"  - 目标帧尺寸: {target_frame_size}")
        print(f"  - 原始尺寸: {original_size}")
        print(f"  - 量化倍数: {quant_multi}")
        
        # 反量化（除以8）
        tracks_dequantized = tracks_np / quant_multi
        print(f"  - 反量化后数据范围: min={tracks_dequantized.min():.2f}, max={tracks_dequantized.max():.2f}")
        
        # 确定数据格式并转换为 [frames, points, 3] 格式
        if len(tracks_dequantized.shape) == 4:
            if tracks_dequantized.shape[0] < tracks_dequantized.shape[1]:
                # [points, frames, batch, 3] -> [frames, points, 3]
                print(f"  - 转换格式: [points, frames, batch, 3] -> [frames, points, 3]")
                tracks_reshaped = tracks_dequantized.transpose(1, 0, 2, 3)  # [frames, points, batch, 3]
                tracks_reshaped = tracks_reshaped.squeeze(2)  # [frames, points, 3]
            else:
                # [frames, channels, points, 3] -> [frames, points, 3]
                print(f"  - 转换格式: [frames, channels, points, 3] -> [frames, points, 3]")
                tracks_reshaped = tracks_dequantized[:, 0, :, :]  # 取第一个channel
        else:
            # [frames, points, 3] 格式
            print(f"  - 保持格式: [frames, points, 3]")
            tracks_reshaped = tracks_dequantized
        
        # 只使用3通道数据 [x, y, visibility]
        tracks_with_time = tracks_reshaped
        
        print(f"  - 最终数据形状: {tracks_with_time.shape}")
        print(f"  - 坐标范围: X=[{tracks_with_time[:, :, 0].min():.1f}, {tracks_with_time[:, :, 0].max():.1f}]")
        print(f"  - 坐标范围: Y=[{tracks_with_time[:, :, 1].min():.1f}, {tracks_with_time[:, :, 1].max():.1f}]")
        print(f"  - 可见性范围: [{tracks_with_time[:, :, 2].min():.1f}, {tracks_with_time[:, :, 2].max():.1f}]")
        
        return torch.from_numpy(tracks_with_time).float()
    
    def extract_frame_with_tracks(self, video_path: str, tracks_data: torch.Tensor, 
                                frame_index: int, estimated_original_size: Tuple[int, int], 
                                selected_point: int = None, highlight_point: int = None):
        """
        提取指定帧并添加轨迹可视化（交互式版本）
        
        Args:
            video_path: 视频文件路径
            tracks_data: 轨迹数据
            frame_index: 帧索引
            estimated_original_size: 原始尺寸
            selected_point: 选中的点ID
            highlight_point: 高亮的点ID
            
        Returns:
            Tuple[frame, error_message, point_info]
        """
        try:
            # 打开视频
            cap = cv2.VideoCapture(video_path)
            
            # 获取视频属性
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 跳转到指定帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return None, f"无法读取帧 {frame_index}", []
            
            # 分离轨迹数据 - [x, y, visibility]格式
            tracks, visible = torch.split(tracks_data, [2, 1], dim=-1)
            
            # 直接使用帧索引（1:1对应）
            track_frame_idx = min(frame_index, len(tracks) - 1)
            
            # 获取当前帧的轨迹点
            current_tracks = tracks[track_frame_idx]  # [points, 2]
            current_visible = visible[track_frame_idx]  # [points, 1]
            
            # 从256x256空间映射到原始视频尺寸
            scale_x = width / 256.0
            scale_y = height / 256.0
            pixel_coords = current_tracks * torch.tensor([scale_x, scale_y]).type_as(current_tracks)
            
            print(f"🔧 坐标映射调试 - 帧 {frame_index}:")
            print(f"   视频尺寸: {width}x{height}")
            print(f"   缩放因子: scale_x={scale_x:.3f}, scale_y={scale_y:.3f}")
            print(f"   256空间坐标范围: X=[{current_tracks[:, 0].min():.1f}, {current_tracks[:, 0].max():.1f}], Y=[{current_tracks[:, 1].min():.1f}, {current_tracks[:, 1].max():.1f}]")
            print(f"   映射后坐标范围: X=[{pixel_coords[:, 0].min():.1f}, {pixel_coords[:, 0].max():.1f}], Y=[{pixel_coords[:, 1].min():.1f}, {pixel_coords[:, 1].max():.1f}]")
            
            # 准备点信息（用于点击检测）
            point_info = []
            
            # 🎨 使用固定的颜色映射，确保与可视化视频一致
            num_points = len(pixel_coords)
            if self.point_colors is None or len(self.point_colors) != num_points:
                # 第一次初始化或点数量变化时重新生成颜色
                self.point_colors = generate_point_colors(num_points)
                print(f"🎨 生成固定颜色映射: {num_points}个点")
            point_colors = self.point_colors
            
            # 绘制轨迹点
            for i in range(num_points):
                x, y = pixel_coords[i].tolist()
                is_visible = current_visible[i, 0] > 0
                
                # 检查点是否在图像边界内
                in_bounds = 0 <= x < width and 0 <= y < height
                
                # 记录点信息
                point_info.append({
                    'id': i,
                    'x': x,
                    'y': y,
                    'visible': is_visible,
                    'in_bounds': in_bounds
                })
                
                # 只绘制在边界内的点
                if in_bounds:
                    # 根据选中状态确定绘制样式
                    if selected_point is not None and i == selected_point:
                        # 选中的点
                        color = point_colors[i] if is_visible else (128, 128, 128)
                        radius = 8
                        border_radius = 10
                        border_thickness = 3
                        border_color = (255, 255, 255)
                    elif highlight_point is not None and i == highlight_point:
                        # 高亮点
                        color = point_colors[i] if is_visible else (160, 160, 160)
                        radius = 6
                        border_radius = 8
                        border_thickness = 2
                        border_color = (255, 255, 255)
                    else:
                        # 普通点
                        if is_visible:
                            color = point_colors[i]
                            radius = 4
                            border_radius = 6
                            border_thickness = 1
                            border_color = (255, 255, 255)
                        else:
                            color = (200, 200, 200)
                            radius = 3
                            border_radius = 5
                            border_thickness = 1
                            border_color = (180, 180, 180)
                    
                    # 绘制边框和内圆
                    cv2.circle(frame, (int(x), int(y)), border_radius, border_color, border_thickness)
                    cv2.circle(frame, (int(x), int(y)), radius, color, -1)
                    
                    # 绘制点ID
                    if selected_point == i or highlight_point == i:
                        text_color = (255, 255, 255)
                    elif is_visible:
                        text_color = (0, 0, 0)
                    else:
                        text_color = (128, 128, 128)
                    
                    cv2.putText(frame, str(i), (int(x) + 12, int(y) - 8), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
            
            # 转换图像格式：BGR -> RGB -> PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            return frame_pil, None, point_info
            
        except Exception as e:
            return None, f"Error extracting frame: {e}", []
    
    def find_nearest_point(self, click_x: float, click_y: float, max_distance: float = 20) -> Optional[int]:
        """
        查找最近的点
        
        Args:
            click_x: 点击的x坐标
            click_y: 点击的y坐标
            max_distance: 最大距离阈值
            
        Returns:
            最近点的ID，如果没有找到则返回None
        """
        min_distance = float('inf')
        nearest_point = None
        
        for point in self.point_info:
            if point['visible'] and point['in_bounds']:
                distance = ((click_x - point['x']) ** 2 + (click_y - point['y']) ** 2) ** 0.5
                if distance < max_distance and distance < min_distance:
                    min_distance = distance
                    nearest_point = point['id']
        
        return nearest_point
    
    def update_point_coordinates(self, point_id: int, new_x: float, new_y: float, frame_index: int) -> Tuple[bool, str]:
        """
        更新点的坐标
        
        Args:
            point_id: 点ID
            new_x: 新的x坐标
            new_y: 新的y坐标
            frame_index: 帧索引
            
        Returns:
            Tuple[success, message]
        """
        try:
            if self.tracks_data is None:
                return False, "请先处理视频和轨迹文件"
            
            # 获取视频尺寸
            cap = cv2.VideoCapture(self.video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            # 分离轨迹数据 - [x, y, visibility]格式
            tracks, visible = torch.split(self.tracks_data, [2, 1], dim=-1)
            
            # 直接使用帧索引
            track_frame_idx = min(frame_index, len(tracks) - 1)
            
            # 检查索引是否在有效范围内
            if point_id >= tracks.shape[1]:
                return False, f"点ID {point_id} 超出范围，最大点数为 {tracks.shape[1]}"
            
            # 从原始视频尺寸映射回256x256空间
            scale_x = 256.0 / width
            scale_y = 256.0 / height
            new_coords_256 = torch.tensor([new_x * scale_x, new_y * scale_y]).type_as(tracks)
            
            print(f"🔧 坐标更新调试:")
            print(f"   原始视频尺寸: {width}x{height}")
            print(f"   UI点击坐标: ({new_x:.1f}, {new_y:.1f})")
            print(f"   映射到256空间: ({new_coords_256[0]:.1f}, {new_coords_256[1]:.1f})")
            
            # 更新轨迹数据（保存256空间的坐标）
            tracks[track_frame_idx, point_id, 0] = new_coords_256[0]
            tracks[track_frame_idx, point_id, 1] = new_coords_256[1]
            
            # 重新组合轨迹数据
            self.tracks_data = torch.cat([tracks, visible], dim=-1)
            
            # 更新原始数据（传递256空间的坐标）
            self._update_original_data_coordinates_256(point_id, new_coords_256[0].item(), new_coords_256[1].item(), track_frame_idx)
            
            return True, f"已更新点 {point_id} 的坐标到 ({new_x:.0f}, {new_y:.0f})"
            
        except Exception as e:
            return False, f"更新坐标时出错: {str(e)}"
    
    def toggle_point_visibility(self, point_id: int, frame_index: int) -> Tuple[bool, str]:
        """
        切换点的可见性
        
        Args:
            point_id: 点ID
            frame_index: 帧索引
            
        Returns:
            Tuple[success, message]
        """
        try:
            if self.tracks_data is None:
                return False, "请先处理视频和轨迹文件"
            
            # 分离轨迹数据 - [x, y, visibility]格式
            tracks, visible = torch.split(self.tracks_data, [2, 1], dim=-1)
            
            # 直接使用帧索引
            track_frame_idx = min(frame_index, len(tracks) - 1)
            
            # 检查索引是否在有效范围内
            if point_id >= tracks.shape[1]:
                return False, f"点ID {point_id} 超出范围，最大点数为 {tracks.shape[1]}"
            
            # 获取当前可见性状态并切换
            current_visibility = visible[track_frame_idx, point_id, 0]
            new_visibility = 1.0 - current_visibility  # 切换：0->1, 1->0
            
            # 更新可见性
            visible[track_frame_idx, point_id, 0] = new_visibility
            
            # 重新组合轨迹数据
            self.tracks_data = torch.cat([tracks, visible], dim=-1)
            
            # 更新原始数据
            self._update_original_data_visibility(point_id, new_visibility, track_frame_idx)
            
            visibility_text = "可见" if new_visibility > 0.1 else "不可见"
            return True, f"已将点 {point_id} 的可见性切换为: {visibility_text}"
            
        except Exception as e:
            return False, f"切换可见性时出错: {str(e)}"
    
    def _update_original_data_coordinates_256(self, point_id: int, coord_256_x: float, coord_256_y: float, 
                                            track_frame_idx: int):
        """更新原始数据中的坐标（输入已经是256空间坐标）"""
        if self.original_raw_data is None:
            return
        
        try:
            # 直接量化256空间的坐标
            orig_x_quantized = coord_256_x * 8
            orig_y_quantized = coord_256_y * 8
            
            print(f"🔧 原始数据更新调试:")
            print(f"   256空间坐标: ({coord_256_x:.1f}, {coord_256_y:.1f})")
            print(f"   量化后坐标: ({orig_x_quantized:.1f}, {orig_y_quantized:.1f})")
            
            # 根据数据格式更新: [points, frames, batch, 3] - (x, y, visibility)
            if len(self.original_raw_data.shape) == 4:
                if self.original_raw_data.shape[0] < self.original_raw_data.shape[1]:
                    # [points, frames, batch, 3] 格式 - (x, y, visibility)
                    self.original_raw_data[point_id, track_frame_idx, 0, 0] = orig_x_quantized  # x坐标在第1位
                    self.original_raw_data[point_id, track_frame_idx, 0, 1] = orig_y_quantized  # y坐标在第2位
                else:
                    # [frames, channels, points, 3] 格式
                    self.original_raw_data[track_frame_idx, :, point_id, 0] = orig_x_quantized
                    self.original_raw_data[track_frame_idx, :, point_id, 1] = orig_y_quantized
            else:
                # [frames, points, 3] 格式
                self.original_raw_data[track_frame_idx, point_id, 0] = orig_x_quantized
                self.original_raw_data[track_frame_idx, point_id, 1] = orig_y_quantized
                
        except Exception as e:
            print(f"❌ 原始数据坐标更新失败: {str(e)}")
    
    def _update_original_data_visibility(self, point_id: int, new_visibility: float, track_frame_idx: int):
        """更新原始数据中的可见性"""
        if self.original_raw_data is None:
            return
        
        try:
            # 量化可见性值
            orig_visibility_quantized = new_visibility * 8
            
            # 根据数据格式更新: [points, frames, batch, 3] - (x, y, visibility)
            if len(self.original_raw_data.shape) == 4:
                if self.original_raw_data.shape[0] < self.original_raw_data.shape[1]:
                    # [points, frames, batch, 3] 格式 - (x, y, visibility)
                    self.original_raw_data[point_id, track_frame_idx, 0, 2] = orig_visibility_quantized  # visibility在第3位
                else:
                    # [frames, channels, points, 3] 格式
                    self.original_raw_data[track_frame_idx, :, point_id, 2] = orig_visibility_quantized
            else:
                # [frames, points, 3] 格式
                self.original_raw_data[track_frame_idx, point_id, 2] = orig_visibility_quantized
                
        except Exception as e:
            print(f"❌ 原始数据可见性更新失败: {str(e)}")
    
    def save_modified_tracks(self, save_path: str = None) -> Tuple[Optional[str], str]:
        """
        保存修改后的轨迹数据
        
        Args:
            save_path: 保存路径，如果为None则自动生成
            
        Returns:
            Tuple[file_path, message]
        """
        try:
            if self.original_raw_data is None:
                return None, "❌ 没有可保存的轨迹数据"
            
            if save_path is None:
                # 自动生成保存路径 - 保存到视频专用文件夹
                video_name = os.path.splitext(os.path.basename(self.video_path))[0]
                from ..core.config import config
                base_output_dir = config.ensure_output_dir()
                video_output_dir = os.path.join(base_output_dir, video_name)
                os.makedirs(video_output_dir, exist_ok=True)
                save_path = os.path.join(video_output_dir, f"{video_name}.pth")
            
            # 压缩并保存数据
            with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as npz_file:
                np.savez_compressed(npz_file, array=self.original_raw_data)
                temp_npz_path = npz_file.name
            
            # 读取压缩数据
            with open(temp_npz_path, 'rb') as f:
                compressed_data = f.read()
            
            # 保存为.pth文件
            torch.save(compressed_data, save_path)
            
            # 清理临时文件
            os.unlink(temp_npz_path)
            
            print(f"✅ 编辑后的轨迹已保存到: {save_path}")
            
            success_msg = f"""✅ 轨迹保存成功!

📁 **文件路径**: `{save_path}`
📊 **数据形状**: {self.original_raw_data.shape}
🎬 **原始视频**: {os.path.basename(self.video_path)}

💡 **提示**: 保存的是编辑后的原始格式数据，可用于重新加载编辑或生成新的可视化视频。
"""
            
            return save_path, success_msg
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            return None, f"❌ 保存轨迹失败: {str(e)}\n详细错误:\n{error_detail}"


# 全局实例（用于向后兼容）
_global_edit_manager = TrackEditManager()


# 向后兼容的函数接口
def extract_frame_with_tracks_interactive(video_path, tracks_data, frame_index, estimated_original_size, 
                                        selected_point=None, highlight_point=None):
    """提取指定帧并添加轨迹可视化（向后兼容函数）"""
    return _global_edit_manager.extract_frame_with_tracks(
        video_path, tracks_data, frame_index, estimated_original_size, selected_point, highlight_point
    )


def find_nearest_point(click_x, click_y, point_info, max_distance=20):
    """查找最近的点（向后兼容函数）"""
    # 更新全局管理器的点信息
    _global_edit_manager.point_info = point_info
    return _global_edit_manager.find_nearest_point(click_x, click_y, max_distance)


def update_point_coordinates(point_id, new_x, new_y, frame_index):
    """更新点的坐标（向后兼容函数）"""
    return _global_edit_manager.update_point_coordinates(point_id, new_x, new_y, frame_index)


def toggle_point_visibility(point_id, frame_index):
    """切换点的可见性（向后兼容函数）"""
    return _global_edit_manager.toggle_point_visibility(point_id, frame_index)


def save_modified_tracks(save_path=None):
    """保存修改后的轨迹数据（向后兼容函数）"""
    return _global_edit_manager.save_modified_tracks(save_path) 