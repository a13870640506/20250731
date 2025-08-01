"""
基于Transformer神经网络的隧道数字孪生系统API
主服务器程序
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import traceback
import numpy as np
import pandas as pd

# 导入自定义模块
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from optimizer import ModelOptimizer
from utils import download_file_handler

# 导入蓝图
from routes import blueprints

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 配置常量
MODEL_DIR = './models'
DATA_DIR = './data'
SCALER_DIR = './scalers'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)

# 创建各模块处理器
data_processor = DataProcessor(data_dir=DATA_DIR, scaler_dir=SCALER_DIR)
model_trainer = ModelTrainer(model_dir=MODEL_DIR, scaler_dir=SCALER_DIR)
model_optimizer = ModelOptimizer(model_dir=MODEL_DIR)

# 注册所有蓝图
for blueprint in blueprints:
    app.register_blueprint(blueprint)

# 首页
@app.route('/')
def home():
    return "基于Transformer神经网络的隧道数字孪生系统API"

# 数据集上传和处理接口
@app.route('/transformer/upload', methods=['POST'])
def upload_dataset():
    try:
        # 获取上传的文件
        input_file = request.files.get('input_file')

        if not input_file:
            return jsonify({
                'success': False,
                'message': '请上传数据文件'
            })

        # 获取数据集划分比例
        train_ratio = float(request.form.get('train_ratio', 0.7))
        val_ratio = float(request.form.get('val_ratio', 0.1))
        test_ratio = float(request.form.get('test_ratio', 0.2))

        # 验证比例之和是否为1
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
            return jsonify({
                'success': False,
                'message': '数据集划分比例之和必须为1'
            })

        # 处理数据集
        result = data_processor.process_data(
            input_file,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )

        return jsonify({
            'success': True,
            'message': '数据处理成功',
            'data': result
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'数据处理失败: {str(e)}'
        })

# 文件下载接口
@app.route('/download', methods=['GET'])
def download_file():
    try:
        file_path = request.args.get('path')
        return download_file_handler(file_path, allowed_dirs=[DATA_DIR, MODEL_DIR])
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'文件下载失败: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)