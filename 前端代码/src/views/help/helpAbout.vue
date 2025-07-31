<script setup>
import { ref } from 'vue'
</script>

<template>
  <div class="help-about">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>关于系统</span>
        </div>
      </template>

      <div class="content">
        <h2>基于Transformer神经网络的隧道数字孪生系统</h2>
        
        <div class="section">
          <h3>系统简介</h3>
          <p>本系统是一个基于Transformer神经网络的隧道数字孪生系统，用于隧道位移预测和分析。系统采用最先进的Transformer架构，能够有效捕捉时间序列数据中的长期依赖关系，提供高精度的预测结果。</p>
        </div>

        <div class="section">
          <h3>主要功能</h3>
          <ul>
            <li><strong>Transformer模型训练</strong>：上传数据集，设置模型参数，训练隧道位移预测模型</li>
            <li><strong>超参数优化</strong>：使用贝叶斯优化方法自动寻找最优模型超参数</li>
            <li><strong>隧道位移预测</strong>：使用训练好的模型进行隧道位移预测</li>
            <li><strong>数据可视化</strong>：对隧道监测数据和预测结果进行可视化展示</li>
          </ul>
        </div>

        <div class="section">
          <h3>技术架构</h3>
          <ul>
            <li><strong>前端</strong>：Vue3 + Vite + Element Plus</li>
            <li><strong>后端</strong>：Python Flask</li>
            <li><strong>深度学习框架</strong>：PyTorch</li>
            <li><strong>优化算法</strong>：Optuna (贝叶斯优化)</li>
          </ul>
        </div>

        <div class="section">
          <h3>使用指南</h3>
          <ol>
            <li>首先在"Transformer模型训练"页面上传数据并训练模型</li>
            <li>或者在"超参数优化"页面寻找最优模型参数</li>
            <li>然后在"隧道位移预测"页面使用训练好的模型进行预测</li>
            <li>最后可以在"数据可视化"页面查看和分析结果</li>
          </ol>
        </div>

        <div class="section">
          <h3>关于Transformer</h3>
          <p>Transformer是一种基于自注意力机制的神经网络架构，最初在自然语言处理领域取得突破性进展。本系统将Transformer应用于时间序列预测，通过自注意力机制捕捉时序数据中的长短期依赖关系，提高预测精度。</p>
        </div>

        <div class="version">
          <p>版本: 1.0.0</p>
          <p>© 2025 隧道数字孪生系统</p>
        </div>
      </div>
    </el-card>
  </div>
</template>

<style scoped lang="scss">
.help-about {
  padding: 20px;

  .card-header {
    font-size: 18px;
    font-weight: bold;
  }

  .content {
    padding: 10px;

    h2 {
      text-align: center;
      margin-bottom: 30px;
      color: #409EFF;
    }

    .section {
      margin-bottom: 30px;

      h3 {
        margin-bottom: 15px;
        padding-bottom: 8px;
        border-bottom: 1px solid #ebeef5;
        color: #303133;
      }

      p {
        line-height: 1.8;
        color: #606266;
        text-indent: 2em;
      }

      ul, ol {
        padding-left: 20px;
        line-height: 1.8;
        color: #606266;

        li {
          margin-bottom: 8px;
        }
      }
    }

    .version {
      margin-top: 40px;
      text-align: center;
      color: #909399;
      font-size: 14px;
    }
  }
}
</style>