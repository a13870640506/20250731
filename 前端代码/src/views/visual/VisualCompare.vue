<script setup>
import { ref, reactive, onMounted } from 'vue'
import { ElMessage, ElLoading } from 'element-plus'

// 模型选择相关
const selectedModelPaths = ref([])
const recentModelPaths = ref([])
const recentModelDates = ref([])

// 加载状态
const isLoading = ref(false)
const isComparing = ref(false)

// 比较结果
const comparisonResult = ref(null)

// 是否显示比较结果
const showComparison = ref(false)

// 参数选择
const selectedParameter = ref('all')
const parameterOptions = ref([
  { label: '全部参数', value: 'all' },
  { label: '拱顶下沉', value: '拱顶下沉' },
  { label: '拱顶下沉2', value: '拱顶下沉2' },
  { label: '周边收敛1', value: '周边收敛1' },
  { label: '周边收敛2', value: '周边收敛2' },
  { label: '拱脚下沉', value: '拱脚下沉' }
])

// 加载最近训练的模型列表
const loadRecentModels = async () => {
  try {
    isLoading.value = true

    // 模拟API调用
    setTimeout(() => {
      recentModelPaths.value = [
        'models/model_c256_lr0.000086_bs16',
        'models/model_c128_lr0.000125_bs32',
        'models/model_c512_lr0.000054_bs8'
      ]
      recentModelDates.value = [
        '2025-07-30 15:30:45',
        '2025-07-29 10:15:22',
        '2025-07-28 09:45:10'
      ]
      isLoading.value = false
    }, 500)
  } catch (error) {
    console.error('获取最近模型列表错误:', error)
    ElMessage.error('获取最近训练记录出错')
    isLoading.value = false
  }
}

// 比较模型
const compareModels = () => {
  if (selectedModelPaths.value.length < 2) {
    ElMessage.warning('请至少选择两个模型进行比较')
    return
  }

  isComparing.value = true

  // 模拟API调用
  setTimeout(() => {
    // 生成模拟的比较结果
    comparisonResult.value = {
      models: selectedModelPaths.value,
      metrics: selectedModelPaths.value.map((path, index) => ({
        model_path: path,
        test_metrics: {
          loss: 0.02 - index * 0.005,
          r2: 0.89 + index * 0.02,
          mape: 7.8 - index * 0.5
        }
      })),
      charts: [
        { param: '拱顶下沉', type: 'comparison', image_path: '' },
        { param: '拱顶下沉2', type: 'comparison', image_path: '' },
        { param: '周边收敛1', type: 'comparison', image_path: '' },
        { param: '周边收敛2', type: 'comparison', image_path: '' },
        { param: '拱脚下沉', type: 'comparison', image_path: '' }
      ]
    }

    showComparison.value = true
    isComparing.value = false
    ElMessage.success('模型比较完成')
  }, 1500)
}

// 格式化数字，根据不同类型的指标使用不同的格式
const formatNumber = (value, type = 'default') => {
  if (value === undefined || value === null) return 'N/A'

  // 将字符串转换为数字
  if (typeof value === 'string') {
    value = parseFloat(value)
  }

  // 如果不是数字，返回原值
  if (isNaN(value)) return value

  // 根据类型格式化
  switch (type) {
    case 'mse': // MSE保留5位小数
      return value.toFixed(5)
    case 'r2': // R2显示为百分比，保留2位小数
      return (value * 100).toFixed(2) + '%'
    case 'mape': // MAPE已经是百分比，保留2位小数
      return value.toFixed(2) + '%'
    default:
      return value.toFixed(6)
  }
}

// 组件挂载时加载最近模型
onMounted(() => {
  loadRecentModels()
})
</script>

<template>
  <div class="visual-compare">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>结果对比</span>
        </div>
      </template>

      <el-row :gutter="20">
        <el-col :span="24">
          <el-card class="model-select-card" shadow="hover">
            <template #header>
              <div class="card-header">
                <span>模型选择</span>
              </div>
            </template>

            <el-form>
              <el-form-item label="选择要比较的模型">
                <el-select v-model="selectedModelPaths" multiple placeholder="选择模型路径" style="width: 100%" filterable
                  :loading="isLoading">
                  <el-option v-for="(path, index) in recentModelPaths" :key="index" :label="path" :value="path">
                    <span class="model-path-option">{{ path }}</span>
                    <span class="model-path-date text-muted">{{ recentModelDates[index] }}</span>
                  </el-option>
                </el-select>
                <div class="form-tip">请选择2-3个模型进行比较</div>
              </el-form-item>

              <el-form-item>
                <el-button type="primary" :loading="isComparing" @click="compareModels" style="width: 100%">
                  比较所选模型
                </el-button>
              </el-form-item>
            </el-form>
          </el-card>
        </el-col>
      </el-row>

      <div v-if="!showComparison" class="no-result">
        <el-empty description="暂无比较结果" />
        <p class="no-result-tip">请选择至少两个模型并点击"比较所选模型"</p>
      </div>

      <div v-else>
        <el-divider>评估指标对比</el-divider>
        <el-table :data="comparisonResult.metrics" border style="width: 100%">
          <el-table-column prop="model_path" label="模型路径" width="300" />
          <el-table-column label="MSE (损失)" width="150">
            <template #default="scope">
              {{ formatNumber(scope.row.test_metrics.loss, 'mse') }}
            </template>
          </el-table-column>
          <el-table-column label="R² 系数" width="150">
            <template #default="scope">
              {{ formatNumber(scope.row.test_metrics.r2, 'r2') }}
            </template>
          </el-table-column>
          <el-table-column label="MAPE" width="150">
            <template #default="scope">
              {{ formatNumber(scope.row.test_metrics.mape, 'mape') }}
            </template>
          </el-table-column>
        </el-table>

        <el-divider>预测结果对比图表</el-divider>
        <div class="chart-filter-container">
          <div class="chart-filter">
            <span class="filter-label">筛选参数：</span>
            <el-select v-model="selectedParameter" placeholder="选择参数" style="width: 200px">
              <el-option v-for="option in parameterOptions" :key="option.value" :label="option.label"
                :value="option.value" />
            </el-select>
          </div>
        </div>

        <div class="charts-container">
          <div v-for="(chart, index) in comparisonResult.charts" :key="index" class="chart-item"
            v-show="selectedParameter === 'all' || selectedParameter === chart.param">
            <div class="chart-placeholder">
              <h3>{{ chart.param }} - 模型对比</h3>
              <p>模型对比图表将显示在这里</p>
              <p>比较模型: {{ comparisonResult.models.join(', ') }}</p>
            </div>
          </div>
        </div>
      </div>
    </el-card>
  </div>
</template>

<style scoped lang="scss">
.visual-compare {
  padding: 15px;

  .card-header {
    font-size: 18px;
    font-weight: bold;
  }

  .model-select-card {
    margin-bottom: 20px;
  }

  .form-tip {
    margin-top: 5px;
    font-size: 12px;
    color: #909399;
    font-style: italic;
  }

  .model-path-option {
    display: inline-block;
    width: 70%;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .model-path-date {
    float: right;
    color: #909399;
    font-size: 12px;
  }

  .no-result {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 400px;

    .no-result-tip {
      margin-top: 20px;
      color: #909399;
    }
  }

  .chart-filter-container {
    display: flex;
    justify-content: center;
    margin: 20px 0;
  }

  .chart-filter {
    display: flex;
    align-items: center;

    .filter-label {
      margin-right: 10px;
      font-weight: bold;
    }
  }

  .charts-container {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
    gap: 20px;
    margin-top: 20px;
  }

  .chart-item {
    border: 1px solid #ebeef5;
    border-radius: 4px;
    overflow: hidden;
  }

  .chart-placeholder {
    width: 100%;
    height: 350px;
    background-color: #f5f7fa;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: #909399;
    font-style: italic;
  }

  .text-muted {
    color: #909399;
  }
}
</style>