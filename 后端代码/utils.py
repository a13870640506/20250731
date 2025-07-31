"""
工具函数模块，提供文件处理等通用功能
"""

import os
import base64
from flask import send_file

def get_base64_image(image_path):
    """
    将图像文件转换为base64编码
    """
    if not os.path.exists(image_path):
        return None
    
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
        return base64.b64encode(image_bytes).decode('utf-8')

def download_file_handler(file_path, allowed_dirs=None):
    """
    处理文件下载，并进行安全检查
    
    参数:
        file_path: 要下载的文件路径
        allowed_dirs: 允许下载的目录列表
        
    返回:
        flask send_file 对象或者错误信息
    """
    if not file_path or not os.path.exists(file_path):
        return {
            'success': False,
            'message': '文件不存在'
        }, 404
    
    # 获取文件名
    filename = os.path.basename(file_path)
    
    # 安全检查：确保只能下载允许的目录下的文件
    if allowed_dirs:
        is_allowed = any(file_path.startswith(directory) for directory in allowed_dirs)
        if not is_allowed:
            return {
                'success': False,
                'message': '无权访问该文件'
            }, 403
    
    # 返回文件
    return send_file(file_path, as_attachment=True, download_name=filename)