"""
模型训练相关路由
对应前端TunnelModel.vue
"""
from flask import request, jsonify
import os
import traceback

from routes import model_training_bp
from model_trainer import ModelTrainer

# 获取模型训练器实例
model_trainer = ModelTrainer(model_dir='./models', scaler_dir='./scalers')

# 模型训练接口
@model_training_bp.route('/train', methods=['POST'])
def train_model_api():
    try:
        # 获取训练参数
        data_path = request.form.get('data_path')
        model_name = request.form.get('model_name', 'transformer')
        epochs = int(request.form.get('epochs', 300))
        batch_size = int(request.form.get('batch_size', 16))
        learning_rate = float(request.form.get('learning_rate', 0.0001))
        weight_decay = float(request.form.get('weight_decay', 0.0001))
        d_model = int(request.form.get('d_model', 256))
        nhead = int(request.form.get('nhead', 4))
        num_layers = int(request.form.get('num_layers', 2))
        mode = request.form.get('mode', 'custom')
        dropout_value = request.form.get('dropout')
        dropout = 0.05  # 默认值
        if dropout_value and dropout_value != 'undefined':
            try:
                dropout = float(dropout_value)
            except ValueError:
                pass  # 如果转换失败，使用默认值
        model_save_path = request.form.get('model_save_path')

        if not data_path or not os.path.exists(data_path):
            return jsonify({
                'success': False,
                'message': '无效的数据路径'
            })

        print(f"开始训练模型，参数：")
        print(f"- 数据路径: {data_path}")
        print(f"- 模型名称: {model_name}")
        print(f"- 训练轮次: {epochs}")
        print(f"- 批次大小: {batch_size}")
        print(f"- 学习率: {learning_rate}")
        print(f"- 权重衰减: {weight_decay}")
        print(f"- 模型维度: {d_model}")
        print(f"- 注意力头数: {nhead}")
        print(f"- Transformer层数: {num_layers}")
        print(f"- Dropout率: {dropout}")
        print(f"- 模型保存路径: {model_save_path}")
        print(f"- 训练模式: {mode}")

        # 调用模型训练器进行训练
        result = model_trainer.train(
            data_path=data_path,
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            model_save_path=model_save_path,
            mode=mode
        )

        return jsonify({
            'success': True,
            'message': '模型训练成功',
            'data': result
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'模型训练失败: {str(e)}'
        })

# 基于文件的模型预测接口
# 为避免与隧道位移预测接口冲突，调整路由路径
@model_training_bp.route('/file_predict', methods=['POST'])
def file_predict():
    try:
        # 获取预测参数
        data_file = request.files.get('data_file')
        model_path = request.form.get('model_path')
        
        if not data_file:
            return jsonify({
                'success': False,
                'message': '请上传预测数据'
            })
        
        if not model_path or not os.path.exists(model_path):
            return jsonify({
                'success': False,
                'message': '无效的模型路径'
            })
        
        # 调用模型训练器进行预测
        result = model_trainer.predict(
            data_file=data_file,
            model_path=model_path
        )
        
        return jsonify({
            'success': True,
            'message': '预测成功',
            'data': result
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'预测失败: {str(e)}'
        })