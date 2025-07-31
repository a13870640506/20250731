# 数字孪生+AI大跨度桥梁抗震与减隔震性能实时预测评估系统

## 项目简介

本系统集成了数字孪生与人工智能技术，面向大跨度桥梁结构的抗震与减隔震性能，提供数据管理、模型训练、地震反应预测评估和数据可视化等一站式服务。系统支持工程师和研究人员对桥梁结构进行实时、智能的性能评估与决策。

![系统概览](https://via.placeholder.com/800x400?text=系统概览图)

## 系统架构

系统采用前后端分离架构：

- **前端**：基于Vue 3开发，使用Element Plus组件库构建用户界面
- **后端**：基于Flask开发的RESTful API服务，集成PyTorch深度学习框架
- **数据流**：用户上传数据 → 模型训练/优化 → 预测评估 → 结果可视化

## 核心功能

### 1. 模型训练与优化
- 支持自定义参数或使用贝叶斯优化进行模型训练
- 自动保存训练过程、结果和模型文件
- 可视化训练损失曲线和评估指标

### 2. 地震响应预测评估
- 上传加速度序列，快速获得位移预测结果
- 自动计算MSE、RMSE、R²等评估指标
- 生成时间序列对比图和散点对比图

### 3. 数据可视化
- 查看保存的预测结果，包含图表和评估指标
- 对比预测值与实际值的表格展示
- 浏览历史数据集，支持筛选和查看

### 4. 系统帮助
- 提供系统使用指南和操作建议
- 常见问题解答
- 版本信息和更新日志

## 技术栈

### 前端
- Vue 3 + Vite
- Element Plus UI组件库
- Vue Router 路由管理
- Axios HTTP客户端
- SCSS样式预处理器

### 后端
- Python 3.8+
- Flask Web框架
- PyTorch 深度学习框架
- PyTorch Lightning 训练框架
- Scikit-learn 机器学习工具
- Matplotlib 数据可视化
- Pandas 数据处理

## 安装部署

### 环境要求
- Node.js 16+
- Python 3.8+
- CUDA 11.3+ (GPU加速，可选)

### 后端部署
1. 克隆代码仓库
   ```bash
   git clone [仓库地址]
   cd [项目目录]/后端代码
   ```

2. 创建并激活Python虚拟环境
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

4. 启动后端服务
   ```bash
   python app.py
   ```
   服务将在 http://localhost:8000 上运行

### 前端部署
1. 进入前端目录
   ```bash
   cd [项目目录]/前端代码
   ```

2. 安装依赖
   ```bash
   npm install
   ```

3. 开发模式运行
   ```bash
   npm run dev
   ```
   或构建生产版本
   ```bash
   npm run build
   ```

4. 访问系统
   开发模式：http://localhost:5173
   生产模式：部署dist目录到Web服务器

## 使用指南

### 基本流程
1. **数据准备**：上传训练集和验证集数据（加速度和位移序列）
2. **模型训练**：选择数据集、设置参数并启动训练
3. **参数优化**：使用贝叶斯优化算法找到最优参数
4. **响应预测**：上传输入文件，选择模型进行预测
5. **结果分析**：查看预测结果、评估指标和可视化图表

### 注意事项
- 数据文件格式必须为CSV，且列名与系统要求一致
- 模型训练和优化可能需要较长时间，请耐心等待
- 预测结果和模型文件会自动保存，可在系统中查看和管理

## 目录结构

```
项目根目录/
├── 前端代码/                  # 前端Vue项目
│   ├── public/               # 静态资源
│   ├── src/                  # 源代码
│   │   ├── api/             # API接口
│   │   ├── assets/          # 资源文件
│   │   ├── components/      # 公共组件
│   │   ├── router/          # 路由配置
│   │   ├── stores/          # 状态管理
│   │   ├── utils/           # 工具函数
│   │   ├── views/           # 页面视图
│   │   ├── App.vue          # 根组件
│   │   └── main.js          # 入口文件
│   └── package.json         # 依赖配置
│
└── 后端代码/                  # 后端Flask项目
    ├── app.py               # 应用入口
    ├── config.py            # 配置文件
    ├── models/              # 模型文件
    ├── results/             # 结果输出
    ├── routes/              # API路由
    ├── services/            # 业务逻辑
    ├── uploads/             # 上传文件
    └── utils/               # 工具函数
```

## 许可证

[MIT License](LICENSE)

## 联系方式

如有问题或建议，请联系项目负责人：[联系邮箱] 