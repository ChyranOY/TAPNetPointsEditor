#!/usr/bin/env python3
"""
File management utilities for TAPNet Tracker.

Contains functions for video file scanning, organization, and metadata management.
"""

import os
import glob
from typing import Dict, List


def scan_video_folder(folder_path: str) -> Dict[str, Dict]:
    """
    扫描文件夹中的视频文件，返回文件信息字典
    
    Args:
        folder_path: 要扫描的文件夹路径
        
    Returns:
        Dict[filename, file_info] 格式的字典
        file_info包含: path, status, result_path, viz_path
    """
    video_files = {}
    
    if not os.path.exists(folder_path):
        print(f"❌ 文件夹不存在: {folder_path}")
        return video_files
    
    # 支持的视频格式
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm']
    
    print(f"🔍 扫描文件夹: {folder_path}")
    
    for ext in video_extensions:
        pattern = os.path.join(folder_path, ext)
        files = glob.glob(pattern)
        files.extend(glob.glob(pattern.upper()))  # 也匹配大写扩展名
        
        for file_path in files:
            filename = os.path.basename(file_path)
            
            # 检查是否已有处理结果
            base_name = os.path.splitext(filename)[0]
            
            # 查找可能的输出文件
            result_path = None
            viz_path = None
            status = 'pending'
            
            # 检查outputs文件夹
            output_dir = "./outputs"
            if os.path.exists(output_dir):
                # 查找轨迹文件
                track_pattern = os.path.join(output_dir, f"{base_name}.pth")
                if os.path.exists(track_pattern):
                    result_path = track_pattern
                    status = 'completed'
                
                # 查找可视化文件
                viz_pattern = os.path.join(output_dir, f"{base_name}_visualization.mp4")
                if os.path.exists(viz_pattern):
                    viz_path = viz_pattern
            
            video_files[filename] = {
                'path': file_path,
                'status': status,
                'result_path': result_path,
                'viz_path': viz_path
            }
    
    print(f"📂 找到 {len(video_files)} 个视频文件")
    for filename, info in video_files.items():
        print(f"  - {filename}: {info['status']}")
    
    return video_files


def format_video_list_display(video_files: Dict[str, Dict]) -> str:
    """
    格式化视频文件列表为显示用的字符串
    
    Args:
        video_files: 视频文件信息字典
        
    Returns:
        格式化的显示字符串
    """
    if not video_files:
        return "📁 未找到视频文件"
    
    display_lines = []
    display_lines.append(f"📂 **共找到 {len(video_files)} 个视频文件:**\n")
    
    # 按状态分组
    pending_files = []
    completed_files = []
    
    for filename, info in video_files.items():
        if info['status'] == 'completed':
            completed_files.append(filename)
        else:
            pending_files.append(filename)
    
    # 显示已完成的文件
    if completed_files:
        display_lines.append("✅ **已处理完成:**")
        for filename in sorted(completed_files):
            display_lines.append(f"  - {filename}")
        display_lines.append("")
    
    # 显示待处理的文件
    if pending_files:
        display_lines.append("⏳ **待处理:**")
        for filename in sorted(pending_files):
            display_lines.append(f"  - {filename}")
        display_lines.append("")
    
    # 添加操作提示
    display_lines.append("💡 **操作说明:**")
    display_lines.append("1. 从下拉菜单中选择要处理的视频")
    display_lines.append("2. 设置轨迹点数和生成方法")
    display_lines.append("3. 点击「生成轨迹」开始处理")
    
    return "\n".join(display_lines)


def get_video_choices(video_files: Dict[str, Dict]) -> List[str]:
    """
    从视频文件字典生成选择列表
    
    Args:
        video_files: 视频文件信息字典
        
    Returns:
        可供选择的视频文件名列表
    """
    if not video_files:
        return ["未找到视频文件"]
    
    choices = []
    
    # 按状态排序，已完成的排在前面
    completed_files = []
    pending_files = []
    
    for filename, info in video_files.items():
        if info['status'] == 'completed':
            completed_files.append(f"✅ {filename}")
        else:
            pending_files.append(f"⏳ {filename}")
    
    # 合并列表
    choices.extend(sorted(completed_files))
    choices.extend(sorted(pending_files))
    
    return choices


def set_output_directory(output_dir: str):
    """设置全局输出目录"""
    global _global_output_dir
    _global_output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)


def get_output_directory() -> str:
    """获取当前输出目录 - 统一使用config中的配置"""
    from ..core.config import config
    return config.ensure_output_dir() 