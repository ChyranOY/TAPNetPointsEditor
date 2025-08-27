#!/usr/bin/env python3
"""
Visualization utilities for TAPNet Tracker.

Contains functions for track visualization, video generation, and color management.
"""

import cv2
import matplotlib
import numpy as np
from typing import List, Tuple


def generate_point_colors(num_points: int) -> List[Tuple[int, int, int]]:
    """生成轨迹点的标准颜色，确保可视化和编辑模式一致
    
    Args:
        num_points: 点的数量
        
    Returns:
        BGR格式的颜色列表
    """
    color_map = matplotlib.colormaps.get_cmap('hsv')
    cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_points - 1)
    point_colors = []
    for i in range(num_points):
        # 获取RGB颜色
        rgb_color = np.array(color_map(cmap_norm(i)))[:3] * 255
        # 转换为BGR格式并确保是整数
        bgr_color = (int(rgb_color[2]), int(rgb_color[1]), int(rgb_color[0]))  # BGR
        point_colors.append(bgr_color)
    return point_colors


def visualize_tracks(frames: np.ndarray, tracks: np.ndarray, visibles: np.ndarray, 
                    tracks_leave_trace: int = 16, fixed_colors: List[Tuple[int, int, int]] = None) -> np.ndarray:
    """可视化轨迹在原始尺寸的视频上"""
    
    num_frames, num_points = tracks.shape[1], tracks.shape[0]
    
    # 🎨 使用固定颜色或生成新颜色
    if fixed_colors is not None and len(fixed_colors) == num_points:
        point_colors = fixed_colors
        print(f"🎨 使用固定颜色映射: {num_points}个点")
    else:
        point_colors = generate_point_colors(num_points)
        print(f"🎨 生成新颜色映射: {num_points}个点")
    
    print(f"开始可视化轨迹:")
    print(f"  - 视频帧数: {num_frames}")
    print(f"  - 轨迹点数: {num_points}")
    print(f"  - 视频尺寸: {frames.shape[1:3]} (height x width)")
    print(f"  - 颜色格式: BGR")
    print(f"  - tracks形状: {tracks.shape}")
    print(f"  - visibles形状: {visibles.shape}")
    
    # 🔍 调试：检查可视化数据的坐标范围 (注意：tracks已翻转)
    print(f"🔍 可视化数据调试:")
    print(f"  - Y坐标范围: [{float(tracks[:, :, 0, 0].min()):.1f}, {float(tracks[:, :, 0, 0].max()):.1f}] (tracks[0])")
    print(f"  - X坐标范围: [{float(tracks[:, :, 0, 1].min()):.1f}, {float(tracks[:, :, 0, 1].max()):.1f}] (tracks[1])")
    print(f"  - 可见性范围: [{float(visibles.min()):.1f}, {float(visibles.max()):.1f}]")
    # 🔍 显示所有点的第一帧详细信息
    print(f"  - 第一帧所有点详情:")
    for i in range(num_points):
        y_raw = float(tracks[i, 0, 0, 0])
        x_raw = float(tracks[i, 0, 0, 1])
        vis = float(visibles[i, 0, 0])
        x_int, y_int = int(round(x_raw)), int(round(y_raw))
        print(f"    点{i}: 原始({x_raw:.2f}, {y_raw:.2f}) 可见性{vis:.2f}")
    
    viz_frames = []
    for t in range(num_frames):
        frame = frames[t].copy()
        
        # 绘制轨迹线
        line_tracks = tracks[:, max(0, t - tracks_leave_trace):t + 1, :]  # [points, time_window, batch, coords]
        line_visibles = visibles[:, max(0, t - tracks_leave_trace):t + 1, 0]  # [points, time_window]
        
        for s in range(line_tracks.shape[1] - 1):
            for i in range(num_points):
                if line_visibles[i, s] > 0.1 and line_visibles[i, s + 1] > 0.1:  # 都可见
                    # 坐标访问修正: tracks[0]是x，tracks[1]是y (与点的访问方式一致)
                    x1, y1 = int(round(line_tracks[i, s, 0, 0])), int(round(line_tracks[i, s, 0, 1]))
                    x2, y2 = int(round(line_tracks[i, s + 1, 0, 0])), int(round(line_tracks[i, s + 1, 0, 1]))
                    
                    # 确保坐标在图像范围内
                    if (0 <= x1 < frame.shape[1] and 0 <= y1 < frame.shape[0] and
                        0 <= x2 < frame.shape[1] and 0 <= y2 < frame.shape[0]):
                        cv2.line(frame, (x1, y1), (x2, y2), point_colors[i], 2, cv2.LINE_AA)
        
        # 绘制当前点
        for i in range(num_points):
            if visibles[i, t, 0] > 0.1:  # 可见
                x, y = int(round(tracks[i, t, 0, 0])), int(round(tracks[i, t, 0, 1]))
                
                # 🔍 调试：输出第一帧的坐标信息
                if t == 0:
                    print(f"  - 点{i}: 坐标({x}, {y}), 可见性{float(visibles[i, t, 0]):.2f}, 视频范围({frame.shape[1]}x{frame.shape[0]})")
                
                # 确保坐标在图像范围内
                if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                    # 绘制彩色内圆
                    cv2.circle(frame, (x, y), 4, point_colors[i], -1, cv2.LINE_AA)
                    # 绘制白色边框
                    cv2.circle(frame, (x, y), 6, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # 🔍 调试：确认绘制了点
                    if t == 0:
                        print(f"    ✅ 点{i}已绘制到({x}, {y})")
                else:
                    # 🔍 调试：坐标超出范围
                    if t == 0:
                        print(f"    ❌ 点{i}坐标({x}, {y})超出视频范围({frame.shape[1]}x{frame.shape[0]})")
        
        viz_frames.append(frame)
    
    return np.stack(viz_frames, axis=0)


def save_visualization_video(frames: np.ndarray, output_path: str, fps: int = 15) -> str:
    """保存可视化视频"""
    # 正确获取视频尺寸 (height, width) -> (width, height)
    height, width = frames.shape[1], frames.shape[2]
    
    print(f"保存可视化视频:")
    print(f"  - 输出路径: {output_path}")
    print(f"  - 视频尺寸: {width}x{height} (width x height)")
    print(f"  - 帧数: {frames.shape[0]}")
    print(f"  - FPS: {fps}")
    
    # 设置视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise ValueError(f"无法创建视频文件: {output_path}")
    
    for frame in frames:
        # 转换RGB到BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"✅ 可视化视频已保存: {output_path}")
    return output_path 