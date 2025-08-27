#!/usr/bin/env python3
"""
Track processing utilities for TAPNet Tracker.

Contains functions for generating query points, processing track data, and saving tracks.
"""

import os
import numpy as np
import torch
from typing import List, Tuple, Optional


def generate_query_points(num_points: int, method: str = "manual", 
                         manual_points: Optional[List[Tuple[float, float]]] = None) -> np.ndarray:
    """
    生成查询点
    
    Args:
        num_points: 兼容性参数（在manual模式下被忽略）
        method: 生成方法 ("manual")
        manual_points: 手动指定的点列表 (归一化坐标，范围0-1)
        
    Returns:
        query_points: shape为[actual_points, 3]的numpy数组，格式为[time, y, x]
    """
    if method == "manual" and manual_points:
        # 处理手动点 (256坐标系) - 恢复原版[visibility, y, x]格式
        click_coords = []
        for x, y in manual_points:
            # TAPNext模型期望坐标格式: [visibility, y, x] - 严格按原版video2track.py实现
            click_coords.append([0.0, float(y), float(x)])  # [visibility, y, x]
        
        query_points = np.array(click_coords, dtype=np.float32)
        
        print(f"🎯 Manual模式: 使用手动选择的 {len(manual_points)} 个点")
        print(f"   原始256坐标: {[(x, y) for x, y in manual_points]}")
        print(f"   转换为[visibility,y,x]: {[(t, y, x) for t, y, x in click_coords]}")
        print(f"   查询点格式: [visibility, y, x] - 严格按原版实现")
        print(f"   查询点形状: {query_points.shape}")
        
        # 添加batch维度 [1, points, 3]
        query_points_with_batch = query_points[None]
        print(f"   添加batch维度后: {query_points_with_batch.shape}")
        return query_points_with_batch
    else:
        raise ValueError("只支持manual模式，请通过点击图像选择轨迹点")


def scale_tracks_to_original_size(tracks: np.ndarray, target_size: Tuple[int, int], 
                                 original_size: Tuple[int, int]) -> np.ndarray:
    """
    将轨迹坐标从目标尺寸缩放到原始尺寸
    
    Args:
        tracks: 轨迹数据，shape为[points, frames, 2]
        target_size: 目标尺寸 (width, height)
        original_size: 原始尺寸 (width, height)
        
    Returns:
        scaled_tracks: 缩放后的轨迹数据
    """
    target_w, target_h = target_size
    original_w, original_h = original_size
    
    # 计算缩放因子
    scale_x = original_w / target_w
    scale_y = original_h / target_h
    
    # 复制轨迹数据以避免修改原始数据
    scaled_tracks = tracks.copy()
    
    # 缩放坐标 - 修正：从调试信息确认[0]是x坐标，[1]是y坐标
    scaled_tracks[:, :, 0] *= scale_x  # x坐标 (tracks[..., 0])
    scaled_tracks[:, :, 1] *= scale_y  # y坐标 (tracks[..., 1])
    
    print(f"Track coordinate scaling:")
    print(f"  - Target size: {target_size}")
    print(f"  - Original size: {original_size}")
    print(f"  - Scale factors: x={scale_x:.2f}, y={scale_y:.2f}")
    print(f"  - Correction: [0]=x coordinate uses scale_x, [1]=y coordinate uses scale_y")
    
    return scaled_tracks


def save_tracks_as_pth(tracks: np.ndarray, visibles: np.ndarray, output_path: str, quant_multi: int = 8):
    """
    将轨迹数据保存为.pth文件，输出格式为[points, frames, batch, 3]
    其中最后一维为(visibility, y, x)
    
    Args:
        tracks: 轨迹数据，shape为[batch, num_points, num_frames, 2] - (x, y)
        visibles: 可见性数据，shape为[batch, num_points, num_frames, 1]
        output_path: 输出文件路径
        quant_multi: 量化因子
    """
    
    print(f"📊 输入数据形状:")
    print(f"  - tracks: {tracks.shape} (batch, points, frames, 2)")
    print(f"  - visibles: {visibles.shape} (batch, points, frames, 1)")
    
    # 保持tracks坐标原始顺序: (x, y)
    # 按照(x, y, visibility)的顺序合并数据
    # tracks: [batch, points, frames, 2] - (x, y)
    # visibles: [batch, points, frames, 1]
    combined_tracks = np.concatenate([tracks, visibles], axis=-1)  # [batch, points, frames, 3]
    
    print(f"📐 合并后数据:")
    print(f"  - 形状: {combined_tracks.shape} (batch, points, frames, 3)")
    print(f"  - 第3维顺序: (x, y, visibility)")
    
    # 转换维度: [batch, points, frames, 3] -> [points, frames, batch, 3]
    combined_tracks = np.transpose(combined_tracks, (1, 2, 0, 3))
    
    # 量化处理
    combined_tracks_quantized = combined_tracks * quant_multi
    combined_tracks_quantized = combined_tracks_quantized.astype(np.float32)
    
    print(f"💾 Final saved data:")
    print(f"  - Data shape: {combined_tracks_quantized.shape} [points, frames, batch, 3]")
    print(f"  - Number of points: {combined_tracks_quantized.shape[0]}")
    print(f"  - Number of frames: {combined_tracks_quantized.shape[1]}")
    print(f"  - Batch count: {combined_tracks_quantized.shape[2]}")
    print(f"  - 3rd dimension: (x, y, visibility)")
    print(f"  - Data range: [{combined_tracks_quantized.min():.2f}, {combined_tracks_quantized.max():.2f}]")
    print(f"  - X coordinate range: [{combined_tracks_quantized[:, :, :, 0].min():.1f}, {combined_tracks_quantized[:, :, :, 0].max():.1f}]")
    print(f"  - Y coordinate range: [{combined_tracks_quantized[:, :, :, 1].min():.1f}, {combined_tracks_quantized[:, :, :, 1].max():.1f}]")
    print(f"  - Visibility range: [{combined_tracks_quantized[:, :, :, 2].min():.1f}, {combined_tracks_quantized[:, :, :, 2].max():.1f}]")
    
    # 创建临时npz文件并保存
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as npz_file:
        npz_path = npz_file.name
    
    # 保存为npz格式
    np.savez_compressed(npz_path, array=combined_tracks_quantized)
    
    # 读取npz文件的字节数据
    with open(npz_path, 'rb') as f:
        compressed_data = f.read()
    
    # 保存为.pth格式
    torch.save(compressed_data, output_path)
    
    # 清理临时npz文件
    os.unlink(npz_path)
    
    print(f"✅ Tracks saved to: {output_path}")
    
    return output_path


def unzip_to_array(data: bytes, key: str = "array") -> np.ndarray:
    """从压缩数据中解压数组"""
    import io
    import tempfile
    import zipfile
    
    if isinstance(data, bytes):
        # 如果是bytes，先保存为临时文件再加载
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp_file:
            tmp_file.write(data)
            tmp_path = tmp_file.name
        
        try:
            # 加载npz文件
            loaded = np.load(tmp_path, allow_pickle=False)
            if hasattr(loaded, 'files'):
                # 如果是npz文件，获取指定的键
                array = loaded[key]
            else:
                # 如果是单个数组
                array = loaded
            return array
        finally:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    else:
        # 如果已经是数组，直接返回
        return data


def scale_coordinates_to_target_size(tracks_coords, original_size, target_size):
    """将坐标从原始尺寸缩放到目标尺寸"""
    original_w, original_h = original_size
    target_w, target_h = target_size
    
    # 计算缩放因子
    scale_x = target_w / original_w
    scale_y = target_h / original_h
    
    # 缩放坐标
    scaled_coords = tracks_coords.copy()
    scaled_coords[:, :, 0] *= scale_x  # x坐标
    scaled_coords[:, :, 1] *= scale_y  # y坐标
    
    return scaled_coords 