#!/usr/bin/env python3
"""
Semantic information management for track points.

Provides functionality to add, edit, and export semantic descriptions for trajectory points.
"""

import os
import json
from datetime import datetime
from typing import Dict, Tuple, Optional, List
from collections import defaultdict

class SemanticInfoManager:
    """Manages semantic information for track points."""
    
    def __init__(self):
        self.semantic_info: Dict[int, str] = {}
        self.num_points: int = 0
        self.video_path: str = None
        self.output_directory: str = "./outputs"
        # 🗂️ 文件路径信息（简化版）
        self.tracks_file_path: str = None   # .pth轨迹文件路径
    
    def initialize(self, num_points: int, video_path: str = None, output_dir: str = None, 
                   copied_video_path: str = None, tracks_file_path: str = None):
        """
        初始化语义信息管理器
        
        Args:
            num_points: 轨迹点数量
            video_path: 视频文件路径
            output_dir: 输出目录
            copied_video_path: 兼容性参数（忽略）
            tracks_file_path: .pth轨迹文件路径
        """
        self.num_points = num_points
        self.video_path = video_path
        self.tracks_file_path = tracks_file_path
        
        if output_dir:
            # 🗂️ 如果提供了视频路径，使用视频专用文件夹
            if video_path:
                import os
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                self.output_directory = os.path.join(output_dir, video_name)
                os.makedirs(self.output_directory, exist_ok=True)
            else:
                self.output_directory = output_dir
        
        # 为每个点初始化空的语义信息
        for point_id in range(num_points):
            if point_id not in self.semantic_info:
                self.semantic_info[point_id] = ""
        
        print(f"📝 初始化语义管理器: {num_points}个点")
        if self.video_path or self.tracks_file_path:
            print(f"🗂️ 文件信息:")
            if self.video_path:
                print(f"  - 视频文件: {os.path.basename(self.video_path)}")
            if self.tracks_file_path:
                print(f"  - 轨迹文件: {os.path.basename(self.tracks_file_path)}")
            print(f"  - 输出目录: {self.output_directory}")
    
    def get_semantic_info(self, point_id: int) -> str:
        """获取指定点的语义信息"""
        return self.semantic_info.get(point_id, "")
    
    def set_semantic_info(self, point_id: int, info: str):
        """设置指定点的语义信息"""
        if 0 <= point_id < self.num_points:
            self.semantic_info[point_id] = info
            return True
        return False
    
    def get_all_semantic_info(self) -> Dict[int, str]:
        """获取所有语义信息"""
        return self.semantic_info.copy()
    
    def save_semantic_info(self, point_id: int, semantic_text: str) -> Tuple[bool, str]:
        """
        保存单个点的语义信息
        
        Args:
            point_id: 点ID
            semantic_text: 语义描述文本
            
        Returns:
            Tuple[success: bool, message: str]
        """
        try:
            if not (0 <= point_id < self.num_points):
                return False, f"❌ 点ID {point_id} 超出有效范围 [0, {self.num_points-1}]"
            
            self.semantic_info[point_id] = semantic_text
            return True, f"✅ 已保存点 {point_id} 的语义信息"
            
        except Exception as e:
            return False, f"❌ 保存失败: {str(e)}"
    
    def get_statistics(self) -> Dict[str, int]:
        """获取语义信息统计"""
        total_points = self.num_points
        filled_points = sum(1 for info in self.semantic_info.values() if info.strip())
        empty_points = total_points - filled_points
        
        return {
            "total_points": total_points,
            "filled_points": filled_points,
            "empty_points": empty_points
        }
    
    def export_semantic_info_to_json(self) -> Tuple[Optional[str], str]:
        """
        📄 导出所有语义信息到JSON文件
        
        Returns:
            Tuple[file_path, message]
        """
        if not self.semantic_info:
            return None, "❌ 没有语义信息可导出"
        
        if self.video_path is None:
            return None, "❌ 没有视频文件信息"
        
        try:
            # 生成输出文件名
            video_name = os.path.splitext(os.path.basename(self.video_path))[0]
            os.makedirs(self.output_directory, exist_ok=True)
            
            semantic_filename = f"{video_name}_semantic_info.json"
            semantic_path = os.path.join(self.output_directory, semantic_filename)
            
            # 构建JSON结构
            export_data = {
                "meta": {
                    "video_file": os.path.basename(self.video_path),
                    "video_name": video_name,
                    "export_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "total_points": self.num_points
                },
                "file_info": {
                    "output_directory": self.output_directory,
                    "semantic_info_file": semantic_filename,
                    "tracks_file_name": os.path.basename(self.tracks_file_path) if self.tracks_file_path else None,
                    "video_file_name": os.path.basename(self.video_path) if self.video_path else None
                },
                "statistics": self.get_statistics(),
                "semantic_info": {}
            }
            
            # 添加语义信息
            for point_id in range(self.num_points):
                semantic_text = self.semantic_info.get(point_id, "")
                export_data["semantic_info"][str(point_id)] = {
                    "point_id": point_id,
                    "point_name": f"Point_{point_id}",
                    "semantic_description": semantic_text,
                    "has_description": bool(semantic_text.strip())
                }
            
            # 保存JSON文件
            with open(semantic_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            stats = self.get_statistics()
            success_msg = f"""✅ **语义信息导出成功！**

📁 **文件路径**: `{semantic_path}`
📊 **统计信息**:
  - 总点数: {stats['total_points']}
  - 已填写: {stats['filled_points']}
  - 未填写: {stats['empty_points']}
📄 **格式**: JSON (便于程序读取和解析)
🗂️ **包含核心文件信息**: 输出目录、文件名等"""
            
            print(f"✅ 语义信息已导出到: {semantic_path}")
            return semantic_path, success_msg
            
        except Exception as e:
            error_msg = f"❌ 导出失败: {str(e)}"
            print(error_msg)
            return None, error_msg
    
    def export_semantic_info_to_txt(self) -> Tuple[Optional[str], str]:
        """
        📄 导出所有语义信息到txt文件 (保留原有格式兼容性)
        
        Returns:
            Tuple[file_path, message]
        """
        if not self.semantic_info:
            return None, "❌ 没有语义信息可导出"
        
        if self.video_path is None:
            return None, "❌ 没有视频文件信息"
        
        try:
            # 生成输出文件名
            video_name = os.path.splitext(os.path.basename(self.video_path))[0]
            os.makedirs(self.output_directory, exist_ok=True)
            
            semantic_filename = f"{video_name}_semantic_info.txt"
            semantic_path = os.path.join(self.output_directory, semantic_filename)
            
            # 构建导出内容
            content_lines = []
            
            # 添加文件头信息
            content_lines.append("# 轨迹点语义信息")
            content_lines.append(f"# 视频文件: {os.path.basename(self.video_path)}")
            content_lines.append(f"# 导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            content_lines.append("")
            
            # 获取统计信息
            stats = self.get_statistics()
            content_lines.append("# 统计信息:")
            content_lines.append(f"# 总点数: {stats['total_points']}")
            content_lines.append(f"# 已填写: {stats['filled_points']}")
            content_lines.append(f"# 未填写: {stats['empty_points']}")
            content_lines.append("")
            content_lines.append("# 语义信息详情:")
            content_lines.append("# 格式: [点ID] 语义描述")
            content_lines.append("")
            
            # 添加所有点的语义信息
            for point_id in range(self.num_points):
                semantic_text = self.semantic_info.get(point_id, "")
                if semantic_text.strip():
                    content_lines.append(f"[{point_id}] {semantic_text}")
                else:
                    content_lines.append(f"[{point_id}] (无描述)")
            
            # 写入文件
            with open(semantic_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(content_lines))
            
            success_msg = f"""✅ **语义信息导出成功！**

📁 **文件路径**: `{semantic_path}`
📊 **统计信息**:
  - 总点数: {stats['total_points']}
  - 已填写: {stats['filled_points']}
  - 未填写: {stats['empty_points']}"""
            
            print(f"✅ 语义信息已导出到: {semantic_path}")
            return semantic_path, success_msg
            
        except Exception as e:
            error_msg = f"❌ 导出失败: {str(e)}"
            print(error_msg)
            return None, error_msg

# 全局语义信息管理器实例
_global_semantic_manager = SemanticInfoManager()

# 全局函数接口（为了保持与现有代码的兼容性）
def initialize_semantic_info(num_points: int = None, video_path: str = None, output_dir: str = None,
                           copied_video_path: str = None, tracks_file_path: str = None):
    """初始化全局语义信息管理器"""
    return _global_semantic_manager.initialize(num_points, video_path, output_dir, None, tracks_file_path)

def get_semantic_info(point_id: int) -> str:
    """获取语义信息"""
    return _global_semantic_manager.get_semantic_info(point_id)

def set_semantic_info(point_id: int, info: str) -> bool:
    """设置语义信息"""
    return _global_semantic_manager.set_semantic_info(point_id, info)

def get_all_semantic_info() -> Dict[int, str]:
    """获取所有语义信息"""
    return _global_semantic_manager.get_all_semantic_info()

def save_point_semantic_info(point_id: int, semantic_text: str) -> Tuple[bool, str]:
    """保存点的语义信息"""
    return _global_semantic_manager.save_semantic_info(point_id, semantic_text)

def export_semantic_info_to_json() -> Tuple[Optional[str], str]:
    """导出语义信息到JSON文件"""
    return _global_semantic_manager.export_semantic_info_to_json()

def export_semantic_info_to_txt() -> Tuple[Optional[str], str]:
    """导出语义信息到txt文件"""
    return _global_semantic_manager.export_semantic_info_to_txt() 