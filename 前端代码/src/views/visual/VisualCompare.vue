<script setup>
import { ref, reactive, onMounted, computed } from 'vue'
import { ElMessage, ElLoading } from 'element-plus'
import { getRecentModelsService, getModelResultService } from '@/api/transformer'
import { baseURL } from '@/utils/request'

// 模型选择相关
const selectedModelPath = ref('')
const recentModelPaths = ref([])
const recentModelDates = ref([])

// 加载状态
const isLoading = ref(false)
const isAnalyzing = ref(false)

// 模型结果数据
const modelResult = ref(null)

// 是否显示分析结果
const showAnalysis = ref(false)

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

// 数据集选择
const selectedDataset = ref('test')
const datasetOptions = [
  { label: '训练集', value: 'train' },
  { label: '验证集', value: 'val' },
  { label: '测试集', value: 'test' }
]

// 对指标进行校验和清洗，确保数值合理
const sanitizeMetrics = (metrics = {}) => {
  const result = {}
  // R2 理论范围为 (-∞, 1]，这里只限制上限
  if (typeof metrics.r2 === 'number') {
    result.r2 = Math.min(metrics.r2, 1)
  }
  // MSE 可能存储在 mse 或 loss 字段中，且不应为负
  const mseVal =
    typeof metrics.mse === 'number'
      ? metrics.mse
      : typeof metrics.loss === 'number'
        ? metrics.loss
        : undefined
  if (typeof mseVal === 'number') {
    result.mse = Math.max(mseVal, 0)
  }
  // MAPE 绝对值不应为负
  if (typeof metrics.mape === 'number') {
    result.mape = Math.max(metrics.mape, 0)
  }
  return result
}

// 误差数据表格
const errorTableData = computed(() => {
  if (!modelResult.value) return []

  const data = []
  const parameters = ['拱顶下沉', '拱顶下沉2', '周边收敛1', '周边收敛2', '拱脚下沉']

  const detailKey = `${selectedDataset.value}_metrics_detail`
  const overallKey = `${selectedDataset.value}_metrics`

  parameters.forEach(param => {
    if (selectedParameter.value === 'all' || selectedParameter.value === param) {
      const metrics = sanitizeMetrics(
        modelResult.value[detailKey]?.[param] || modelResult.value[overallKey] || {}
      )

      data.push({
        parameter: param,
        r2: metrics.r2 ?? 'N/A',
        mse: metrics.mse ?? 'N/A',
        mape: metrics.mape ?? 'N/A'
      })
    }
  })

  if (selectedParameter.value === 'all') {
    const overallMetrics = sanitizeMetrics(modelResult.value[overallKey] || {})

    data.push({
      parameter: '总体',
      r2: overallMetrics.r2 ?? 'N/A',
      mse: overallMetrics.mse ?? 'N/A',
      mape: overallMetrics.mape ?? 'N/A'
    })
  }

  return data
})

// 当前数据集标签
const datasetLabel = computed(() => {
  const opt = datasetOptions.find(o => o.value === selectedDataset.value)
  return opt ? `${opt.label}` : ''
})

// 加载最近训练的模型列表
const loadRecentModels = async () => {
  try {
    isLoading.value = true

    // 调用后端API获取最近模型列表
    const response = await getRecentModelsService()

    if (response.data && response.data.success) {
      recentModelPaths.value = response.data.data.paths || []
      recentModelDates.value = response.data.data.dates || []

      console.log('获取到模型列表:', recentModelPaths.value)

      if (recentModelPaths.value.length > 0 && !selectedModelPath.value) {
        selectedModelPath.value = recentModelPaths.value[0]
        console.log('默认选择模型路径:', selectedModelPath.value)
      } else if (recentModelPaths.value.length === 0) {
        ElMessage.warning('未找到任何模型记录，请手动输入模型路径')
        selectedModelPath.value = ''
      }
    } else {
      ElMessage.warning(response.data?.message || '获取模型列表失败')
      selectedModelPath.value = ''
    }

    isLoading.value = false
  } catch (error) {
    console.error('获取最近模型列表错误:', error)
    ElMessage.error('获取最近训练记录出错')
    isLoading.value = false
    selectedModelPath.value = ''
  }
}

// 分析模型结果
const analyzeModel = async () => {
  if (!selectedModelPath.value) {
    ElMessage.warning('请先选择一个模型')
    return
  }

  isAnalyzing.value = true

  try {
    const loading = ElLoading.service({
      lock: true,
      text: '正在分析模型结果...',
      background: 'rgba(0, 0, 0, 0.7)'
    })

    // 获取模型结果
    const response = await getModelResultService(selectedModelPath.value)

    if (response.data && response.data.success) {
      modelResult.value = response.data.data

      showAnalysis.value = true
      ElMessage.success('模型分析完成')
    } else {
      ElMessage.warning(response.data?.message || '获取模型结果失败')
    }

    loading.close()
  } catch (error) {
    console.error('分析模型错误:', error)
    ElMessage.error('分析模型出错')
  } finally {
    isAnalyzing.value = false
  }
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

// 获取图表URL
const getImageUrl = (modelPath, paramName, datasetType) => {
  // 构建API URL以获取图片
  const baseApiUrl = `${import.meta.env.VITE_API_URL || baseURL}/transformer/model_image?path=`

  // 处理验证集的特殊情况
  if (datasetType === 'val') {
    // 验证集使用测试集的图表格式
    return `${baseApiUrl}${modelPath}&file=test_prediction_${paramName}_zh.png`
  }

  return `${baseApiUrl}${modelPath}&file=${datasetType}_prediction_${paramName}_zh.png`
}

// 获取误差图表URL
const getErrorImageUrl = (modelPath, paramName, datasetType) => {
  // 构建API URL以获取误差图片
  const baseApiUrl = `${import.meta.env.VITE_API_URL || baseURL}/transformer/model_image?path=`

  // 处理验证集的特殊情况
  if (datasetType === 'val') {
    // 验证集使用测试集的误差图表格式
    return `${baseApiUrl}${modelPath}&file=${paramName}_test_errors_zh.png`
  }

  return `${baseApiUrl}${modelPath}&file=${paramName}_${datasetType}_errors_zh.png`
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
        <el-col :span="6">
          <el-card class="model-select-card" shadow="hover">
            <template #header>
              <div class="card-header">
                <span>模型选择</span>
              </div>
            </template>

            <el-divider>选择模型</el-divider>
            <div class="recent-models">
              <el-select v-model="selectedModelPath" placeholder="选择模型路径" style="width: 100%; margin-bottom: 10px;"
                filterable :loading="isLoading">
                <el-option v-for="(path, index) in recentModelPaths" :key="index" :label="path" :value="path">
                  <span class="model-path-option">{{ path }}</span>
                  <span class="model-path-date text-muted">{{ recentModelDates[index] }}</span>
                </el-option>
              </el-select>
            </div>

            <el-button type="primary" :loading="isAnalyzing" @click="analyzeModel"
              style="width: 100%; margin-top: 15px;">
              分析模型结果
            </el-button>

            <el-divider>筛选选项</el-divider>
            <div class="filter-section">
              <div class="filter-item">
                <span class="filter-label">参数选择：</span>
                <el-select v-model="selectedParameter" placeholder="选择参数" style="width: 100%">
                  <el-option v-for="option in parameterOptions" :key="option.value" :label="option.label"
                    :value="option.value" />
                </el-select>
              </div>

              <div class="filter-item" style="margin-top: 15px;">
                <span class="filter-label">数据集选择：</span>
                <el-radio-group v-model="selectedDataset">
                  <el-radio v-for="option in datasetOptions" :key="option.value" :label="option.value">
                    {{ option.label }}
                  </el-radio>
                </el-radio-group>
              </div>
            </div>
          </el-card>
        </el-col>

        <el-col :span="18">
          <div v-if="!showAnalysis" class="no-result">
            <el-empty description="暂无分析结果" />
            <p class="no-result-tip">请在左侧选择一个模型并点击"分析模型结果"</p>
          </div>

          <div v-else>
            <el-divider>误差指标对比</el-divider>

            <!-- 误差对比表格 - 紧凑版 -->
            <div class="compact-table-container">
              <el-table :data="errorTableData" border size="small" style="width: 100%; margin-bottom: 20px;">
                <el-table-column prop="parameter" label="参数" width="90" fixed="left" align="center" />

                <el-table-column :label="datasetLabel" align="center">
                  <el-table-column label="R²" width="80" align="center">
                    <template #default="scope">
                      {{ formatNumber(scope.row.r2, 'r2') }}
                    </template>
                  </el-table-column>
                  <el-table-column label="MSE" width="80" align="center">
                    <template #default="scope">
                      {{ formatNumber(scope.row.mse, 'mse') }}
                    </template>
                  </el-table-column>
                  <el-table-column label="MAPE" width="80" align="center">
                    <template #default="scope">
                      {{ formatNumber(scope.row.mape, 'mape') }}
                    </template>
                  </el-table-column>
                </el-table-column>
              </el-table>
            </div>

            <el-divider>预测结果与误差分析</el-divider>

            <div class="charts-container">
              <template v-for="param in parameterOptions.slice(1)" :key="param.value">
                <div class="chart-item" v-show="selectedParameter === 'all' || selectedParameter === param.value">
                  <h3 class="chart-title">{{ param.label }}</h3>

                  <div class="chart-grid">
                    <!-- 预测结果图表 -->
                    <div class="chart-panel">
                      <div class="chart-image">
                        <img :src="getImageUrl(selectedModelPath, param.value, selectedDataset)"
                          :alt="`${param.label} - ${selectedDataset}`" />
                      </div>
                    </div>

                    <!-- 误差分析图表 -->
                    <div class="chart-panel">
                      <div class="chart-image">
                        <img :src="getErrorImageUrl(selectedModelPath, param.value, selectedDataset)"
                          :alt="`${param.label} - ${selectedDataset}`" />
                      </div>
                    </div>
                  </div>
                </div>
              </template>
            </div>
          </div>
        </el-col>
      </el-row>
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
    height: 100%;
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

  .filter-section {
    margin-top: 15px;
  }

  .filter-item {
    margin-bottom: 15px;
  }

  .filter-label {
    display: block;
    margin-bottom: 8px;
    font-weight: bold;
    color: #606266;
  }

  .compact-table-container {
    overflow-x: auto;

    :deep(.el-table) {
      .el-table__header th {
        padding: 6px 0;
        font-size: 13px;
      }

      .el-table__body td {
        padding: 6px 0;
      }

      .cell {
        padding-left: 5px;
        padding-right: 5px;
      }
    }
  }

  .charts-container {
    display: flex;
    flex-direction: column;
    gap: 30px;
    margin-top: 20px;
  }

  .chart-item {
    border: 1px solid #ebeef5;
    border-radius: 4px;
    padding: 15px;
    background-color: #fff;
    box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.05);
  }

  .chart-title {
    margin-top: 0;
    margin-bottom: 15px;
    font-size: 16px;
    color: #303133;
    text-align: center;
  }

  .chart-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 20px;
  }

  .chart-panel {
    border: 1px solid #ebeef5;
    border-radius: 4px;
    padding: 10px;
  }

  .chart-image {
    width: 100%;
    display: flex;
    justify-content: center;

    img {
      max-width: 100%;
      max-height: 300px;
      border: 1px solid #ebeef5;
      border-radius: 4px;
    }
  }

  .text-muted {
    color: #909399;
  }

  .highlight-best {
    color: #67C23A;
    font-weight: bold;
  }
}
</style>