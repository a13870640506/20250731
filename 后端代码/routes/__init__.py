"""
路由蓝图包初始化文件
"""
from flask import Blueprint

# 创建蓝图
model_training_bp = Blueprint('model_training', __name__, url_prefix='/transformer')
param_optimization_bp = Blueprint('param_optimization', __name__, url_prefix='/transformer')
visualization_bp = Blueprint('visualization', __name__, url_prefix='/transformer')

# 导入路由模块
from routes import model_training, param_optimization, visualization

# 蓝图列表，用于在app.py中注册
blueprints = [
    model_training_bp,
    param_optimization_bp,
    visualization_bp
]