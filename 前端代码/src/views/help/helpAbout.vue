<script setup>
import { ref, onMounted } from 'vue'
import { ArrowDown, Connection, DataAnalysis, Monitor, Setting, TrendCharts, InfoFilled, Document, Edit, UploadFilled, Cpu } from '@element-plus/icons-vue'

// 系统版本信息
const systemInfo = {
  version: 'v1.00',
}

// 技术栈信息
const techStack = [
  { name: '前端框架', value: 'Vue 3 + Element Plus' },
  { name: '后端框架', value: 'Flask' },
  { name: '深度学习框架', value: 'PyTorch' },
  { name: '核心算法', value: 'Transformer神经网络' },
  { name: '优化算法', value: 'Optuna贝叶斯优化' },
  { name: '数据处理', value: 'Pandas + NumPy' },
  { name: '可视化', value: 'Matplotlib + ECharts' }
]

// 功能模块介绍
const features = ref([
  {
    icon: 'UploadFilled',
    title: '数据处理',
    description: '上传CSV数据，自动划分训练集、验证集和测试集',
    color: '#1890ff'
  },
  {
    icon: 'Setting',
    title: '参数优化',
    description: '贝叶斯优化算法自动搜索最佳超参数组合',
    color: '#52c41a'
  },
  {
    icon: 'Monitor',
    title: '模型预测',
    description: '输入围岩参数，预测隧道位移指标',
    color: '#722ed1'
  }
])

// 控制功能卡片的展开状态
const activeFeature = ref('all')

// 切换功能卡片的展开状态
const toggleFeature = (feature) => {
  if (activeFeature.value === feature) {
    activeFeature.value = 'all'
  } else {
    activeFeature.value = feature
  }
}

// 粒子动画
const particlesLoaded = ref(false)

// 模拟粒子加载完成
onMounted(() => {
  setTimeout(() => {
    particlesLoaded.value = true
  }, 500)
})
</script>

<template>
  <div class="help-about">
    <div class="particles-container">
      <div class="particles" :class="{ 'particles-loaded': particlesLoaded }"></div>
    </div>

    <el-row :gutter="20">
      <!-- 左侧系统信息 -->
      <el-col :span="8">
        <el-card class="system-info-card" shadow="hover">
          <div class="system-header">
            <div class="system-logo">
              <div class="logo-animation">
                <el-icon class="logo-icon">
                  <Connection />
                </el-icon>
              </div>
              <h2>DTAI-TE-RTPSP</h2>
            </div>
            <div class="system-title">
              数字孪生+AI交通隧道衬砌受力性能
              <br></br>
              实时预测系统
            </div>
            <div class="system-version">
              <span>版本: {{ systemInfo.version }}</span>
            </div>
          </div>

          <el-divider content-position="center">
            <el-icon style="margin-right: 8px;">
              <Cpu />
            </el-icon>技术栈
          </el-divider>

          <div class="tech-stack">
            <el-descriptions :column="1" border size="small">
              <el-descriptions-item v-for="(tech, index) in techStack" :key="index" :label="tech.name">
                {{ tech.value }}
              </el-descriptions-item>
            </el-descriptions>
          </div>

          <div class="system-intro">
            <el-divider content-position="center">
              <el-icon style="margin-right: 8px;">
                <InfoFilled />
              </el-icon>系统介绍
            </el-divider>
            <p>
              本系统是一个基于Transformer神经网络的智能平台，旨在通过数字孪生技术实现隧道工程的智能化分析与预测。系统集成了数据处理、模型训练、参数优化和可视化分析等功能，为隧道工程提供全方位的数字化解决方案。
            </p>
          </div>
        </el-card>
      </el-col>

      <!-- 右侧功能介绍 -->
      <el-col :span="16">
        <el-card class="feature-card" shadow="hover">
          <template #header>
            <div class="card-header">
              <div class="header-title">
                <el-icon style="margin-right: 10px; font-size: 20px;">
                  <TrendCharts />
                </el-icon>
                <span>系统功能介绍</span>
              </div>
              <div class="header-actions">
                <el-button type="primary" plain size="small" @click="activeFeature = 'all'">展开全部</el-button>
              </div>
            </div>
          </template>

          <!-- 数据处理 -->
          <div class="feature-section" :class="{ active: activeFeature === 'data' }">
            <div class="feature-item-card" @click="toggleFeature('data')">
              <div class="feature-icon-circle" style="background-color: #1890ff;">
                <el-icon>
                  <UploadFilled />
                </el-icon>
              </div>
              <div class="feature-content">
                <h3>数据处理</h3>
                <p>上传CSV数据，自动划分训练集、验证集和测试集</p>
              </div>
              <el-icon class="expand-icon" :class="{ 'is-rotate': activeFeature === 'data' }">
                <ArrowDown />
              </el-icon>
            </div>

            <div class="feature-detail"
              :style="{ maxHeight: activeFeature === 'data' || activeFeature === 'all' ? '500px' : '0', opacity: activeFeature === 'data' || activeFeature === 'all' ? '1' : '0' }">
              <div class="detail-card">
                <p>系统支持上传CSV格式的数据文件，包含输入特征（如泊松比、内摩擦角、粘聚力等）和输出标签（如拱顶下沉、周边收敛等）。上传后系统会自动进行数据处理，包括：</p>
                <ul>
                  <li><span class="highlight">数据集划分：</span>按照设定比例划分训练集、验证集和测试集</li>
                  <li><span class="highlight">数据标准化：</span>对输入特征和输出标签进行标准化处理</li>
                  <li><span class="highlight">日志输出：</span>显示数据处理过程和结果</li>
                </ul>
              </div>
            </div>
          </div>

          <!-- 参数优化 -->
          <div class="feature-section" :class="{ active: activeFeature === 'param' }">
            <div class="feature-item-card" @click="toggleFeature('param')">
              <div class="feature-icon-circle" style="background-color: #52c41a;">
                <el-icon>
                  <Setting />
                </el-icon>
              </div>
              <div class="feature-content">
                <h3>参数优化</h3>
                <p>贝叶斯优化算法自动搜索最佳超参数组合</p>
              </div>
              <el-icon class="expand-icon" :class="{ 'is-rotate': activeFeature === 'param' }">
                <ArrowDown />
              </el-icon>
            </div>

            <div class="feature-detail"
              :style="{ maxHeight: activeFeature === 'param' || activeFeature === 'all' ? '500px' : '0', opacity: activeFeature === 'param' || activeFeature === 'all' ? '1' : '0' }">
              <div class="detail-card">
                <p>系统采用Optuna贝叶斯优化算法，自动搜索最佳的模型超参数组合，包括：</p>
                <ul>
                  <li><span class="highlight">模型维度(d_model)：</span>Transformer模型的嵌入维度</li>
                  <li><span class="highlight">注意力头数(nhead)：</span>多头自注意力机制中的头数</li>
                  <li><span class="highlight">Transformer层数：</span>Transformer编码器层数</li>
                  <li><span class="highlight">学习率：</span>参数更新步长</li>
                  <li><span class="highlight">权重衰减：</span>L2正则化系数</li>
                </ul>
              </div>
            </div>
          </div>

          <!-- 模型预测 -->
          <div class="feature-section" :class="{ active: activeFeature === 'predict' }">
            <div class="feature-item-card" @click="toggleFeature('predict')">
              <div class="feature-icon-circle" style="background-color: #722ed1;">
                <el-icon>
                  <Monitor />
                </el-icon>
              </div>
              <div class="feature-content">
                <h3>模型预测</h3>
                <p>输入围岩参数，预测隧道位移指标</p>
              </div>
              <el-icon class="expand-icon" :class="{ 'is-rotate': activeFeature === 'predict' }">
                <ArrowDown />
              </el-icon>
            </div>

            <div class="feature-detail"
              :style="{ maxHeight: activeFeature === 'predict' || activeFeature === 'all' ? '500px' : '0', opacity: activeFeature === 'predict' || activeFeature === 'all' ? '1' : '0' }">
              <div class="detail-card">
                <p>基于训练好的Transformer模型，输入围岩物理参数，预测隧道各部位位移：</p>
                <ul>
                  <li><span class="highlight">输入参数：</span>泊松比、内摩擦角、粘聚力、剪胀角、弹性模量</li>
                  <li><span class="highlight">预测指标：</span>拱顶下沉、周边收敛、拱脚下沉位移值</li>
                  <li><span class="highlight">结果评估：</span>提供预测结果的可靠性评估和误差分析</li>
                </ul>
              </div>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped lang="scss">
.help-about {
  padding: 10px;
  position: relative;
  overflow: hidden;

  .particles-container {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -1;
    overflow: hidden;

    .particles {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background-image:
        radial-gradient(circle at 25% 25%, rgba(64, 158, 255, 0.05) 1px, transparent 1px),
        radial-gradient(circle at 75% 75%, rgba(64, 158, 255, 0.05) 1px, transparent 1px);
      background-size: 30px 30px;
      opacity: 0;
      transition: opacity 1s ease;

      &.particles-loaded {
        opacity: 1;
      }
    }
  }

  .system-info-card {
    height: 100%;
    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    border-radius: 10px;
    overflow: hidden;
    position: relative;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);

    &::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 5px;
      background: linear-gradient(90deg, #1890ff, #52c41a, #722ed1);
    }

    .system-header {
      text-align: center;
      margin-bottom: 10px;
      padding-top: 10px;

      .system-logo {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 10px;

        .logo-animation {
          position: relative;
          width: 40px;
          height: 40px;
          margin-right: 10px;

          &::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 100%;
            height: 100%;
            border-radius: 50%;
            background-color: rgba(64, 158, 255, 0.1);
            transform: translate(-50%, -50%);
            animation: pulse 2s infinite;
          }

          .logo-icon {
            position: relative;
            font-size: 28px;
            color: #409EFF;
            z-index: 1;
          }
        }

        h2 {
          margin: 0;
          color: #303133;
          font-size: 22px;
          background: linear-gradient(90deg, #1890ff, #52c41a);
          -webkit-background-clip: text;
          background-clip: text;
          -webkit-text-fill-color: transparent;
          font-weight: 600;
        }
      }

      .system-title {
        font-size: 16px;
        color: #606266;
        margin-bottom: 10px;
        line-height: 1.4;
      }

      .system-version {
        display: flex;
        flex-direction: column;
        font-size: 13px;
        color: #909399;

        span {
          margin: 2px 0;
        }
      }
    }

    .tech-stack {
      margin-bottom: 10px;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.05);

      :deep(.el-descriptions__body) {
        .el-descriptions__label {
          padding: 5px 8px;
          font-size: 13px;
          background-color: rgba(64, 158, 255, 0.05);
        }

        .el-descriptions__content {
          padding: 5px 8px;
          font-size: 13px;
        }
      }
    }

    .system-intro {
      margin-top: 10px;

      p {
        font-size: 18px;
        color: #303133;
        line-height: 1.8;
        text-align: justify;
        margin: 2px;
        padding: 10px;
        background-color: rgba(64, 158, 255, 0.05);
        border-radius: 8px;
        border-left: 3px solid #409EFF;
      }
    }
  }

  .feature-card {
    height: 100%;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
    background: linear-gradient(to bottom, #ffffff, #f9fafc);
    border: 1px solid #ebeef5;

    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 16px 20px !important;
      background: linear-gradient(135deg, #f0f5ff, #ffffff);
      border-bottom: 1px solid #ebeef5;

      .header-title {
        display: flex;
        align-items: center;
        font-size: 18px;
        font-weight: 600;
        color: #303133;

        .el-icon {
          margin-right: 12px;
          font-size: 24px;
          color: #409EFF;
          background-color: #ecf5ff;
          padding: 8px;
          border-radius: 8px;
          box-shadow: 0 2px 8px rgba(64, 158, 255, 0.2);
        }
      }

      .header-actions {
        .el-button {
          border-radius: 20px;
          padding: 8px 16px;
          font-weight: 500;
          box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
        }
      }
    }

    .feature-section {
      margin-bottom: 16px;
      border-radius: 12px;
      overflow: hidden;
      transition: all 0.3s ease-in-out;
      box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.03);
      border: 1px solid #f0f0f0;

      &:last-child {
        margin-bottom: 0;
      }

      &.active {
        box-shadow: 0 8px 24px 0 rgba(0, 0, 0, 0.08);
        transform: translateY(-2px);
        border-color: #e6e6e6;

        .feature-item-card {
          background-color: #f9fafc;

          .feature-content {
            h3 {
              color: #303133;
              font-weight: 600;
            }
          }
        }
      }

      .feature-item-card {
        display: flex;
        align-items: center;
        background-color: white;
        padding: 16px 20px;
        border-radius: 12px 12px 0 0;
        position: relative;
        cursor: pointer;
        transition: all 0.3s ease;

        &:hover {
          background-color: #f9fafc;

          .expand-icon {
            color: #409EFF;
          }
        }

        .feature-icon-circle {
          color: white;
          width: 48px;
          height: 48px;
          border-radius: 12px;
          display: flex;
          align-items: center;
          justify-content: center;
          margin-right: 16px;
          box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
          transition: transform 0.3s ease;

          .el-icon {
            font-size: 22px;
          }
        }

        &:hover .feature-icon-circle {
          transform: scale(1.05);
        }

        .feature-content {
          flex: 1;

          h3 {
            margin: 0 0 6px 0;
            font-size: 18px;
            color: #303133;
            transition: color 0.3s ease;
            font-weight: 500;
          }

          p {
            margin: 0;
            font-size: 14px;
            color: #606266;
            transition: color 0.3s ease;
            line-height: 1.5;
          }
        }

        .expand-icon {
          color: #909399;
          font-size: 16px;
          transition: transform 0.4s ease, color 0.3s ease;
          margin-left: 12px;
          background-color: #f5f7fa;
          width: 24px;
          height: 24px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;

          &.is-rotate {
            transform: rotate(180deg);
            color: #409EFF;
            background-color: #ecf5ff;
          }
        }
      }

      .feature-detail {
        background-color: #f9fafc;
        padding: 0;
        border-radius: 0 0 12px 12px;
        font-size: 16px;
        border-top: 1px solid #ebeef5;
        overflow: hidden;
        transition: all 0.3s ease;
        max-height: 0;
        opacity: 0;

        &.v-enter-active,
        &.v-leave-active {
          transition: max-height 0.5s ease, opacity 0.3s ease;
        }

        &.v-enter-from,
        &.v-leave-to {
          max-height: 0;
          opacity: 0;
        }

        &.v-enter-to,
        &.v-leave-from {
          max-height: 500px;
          opacity: 1;
        }

        .detail-card {
          padding: 20px 24px;
          background: linear-gradient(to bottom, #f9fafc, #ffffff);
          box-shadow: inset 0 4px 8px -4px rgba(0, 0, 0, 0.03);

          p {
            margin-top: 0;
            margin-bottom: 16px;
            color: #606266;
            line-height: 1.6;
            font-size: 15px;
          }

          ul {
            margin: 0;
            padding-left: 10px;
            list-style-type: none;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            grid-gap: 10px 20px;

            li {
              margin-bottom: 10px;
              color: #606266;
              position: relative;
              padding-left: 20px;
              font-size: 15px;
              line-height: 1.6;
              display: flex;
              align-items: flex-start;

              &::before {
                content: '';
                position: absolute;
                left: 0;
                top: 8px;
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background-color: #409EFF;
                box-shadow: 0 0 0 2px rgba(64, 158, 255, 0.2);
              }

              .highlight {
                color: #303133;
                font-weight: 600;
                margin-right: 4px;
              }
            }
          }
        }
      }
    }
  }
}

@keyframes pulse {
  0% {
    transform: translate(-50%, -50%) scale(1);
    opacity: 0.6;
  }

  50% {
    transform: translate(-50%, -50%) scale(1.2);
    opacity: 0.3;
  }

  100% {
    transform: translate(-50%, -50%) scale(1);
    opacity: 0.6;
  }
}
</style>