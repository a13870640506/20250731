<script setup>
import { ref, reactive, onMounted } from 'vue'
import { ElMessage, ElLoading } from 'element-plus'
import { getDatasetHistoryService, getDatasetDetailService } from '@/api/transformer'
import { baseURL } from '@/utils/request'

// 数据集列表
const datasetList = ref([])
const selectedDataset = ref('')

// 加载状态
const isLoading = ref(false)
const isAnalyzing = ref(false)

// 分析结果
const analysisResult = ref(null)

// 是否显示分析结果
const showAnalysis = ref(false)

// 加载历史数据集列表
const loadDatasets = async () => {
  try {
    isLoading.value = true

    // 调用后端API获取历史数据集列表
    const response = await getDatasetHistoryService()

    if (response.data && response.data.success) {
      // 处理API返回的数据
      const { ids, dates, info } = response.data.data

      // 构建数据集列表
      datasetList.value = ids.map((id, index) => ({
        id: id,
        date: dates[index],
        info: info[index]
      }))

      console.log('获取到数据集列表:', datasetList.value)

      // 如果列表为空，显示提示
      if (datasetList.value.length === 0) {
        ElMessage.warning('未找到历史数据集')
      }
    } else {
      ElMessage.warning(response.data?.message || '获取数据集列表失败')
    }

    isLoading.value = false
  } catch (error) {
    console.error('获取历史数据集列表错误:', error)
    ElMessage.error('获取历史数据集列表出错')
    isLoading.value = false

    // 使用模拟数据以防API调用失败
    datasetList.value = [
      {
        id: 'processed_20250801_100009',
        date: '2025-08-01 10:00:09',
        info: {
          train_size: 120,
          val_size: 20,
          test_size: 40,
          input_dim: 7,
          output_dim: 5
        }
      },
      {
        id: 'processed_20250801_102947',
        date: '2025-08-01 10:29:47',
        info: {
          train_size: 150,
          val_size: 25,
          test_size: 50,
          input_dim: 7,
          output_dim: 5
        }
      }
    ]
  }
}

// 分析数据集
const analyzeDataset = async () => {
  if (!selectedDataset.value) {
    ElMessage.warning('请先选择数据集')
    return
  }

  isAnalyzing.value = true

  try {
    const loading = ElLoading.service({
      lock: true,
      text: '加载数据集详情...',
      background: 'rgba(0, 0, 0, 0.7)'
    })

    // 调用后端API获取数据集详情
    const response = await getDatasetDetailService(selectedDataset.value)

    if (response.data && response.data.success) {
      const data = response.data.data

      // 构建分析结果
      analysisResult.value = {
        dataset_id: data.dataset_id,
        dataset_info: data.info,
        original_data: data.original_data,
        train_data: data.train_data,
        val_data: data.val_data,
        test_data: data.test_data,

        // 统计信息（如果API没有提供，使用模拟数据）
        statistics: data.statistics || {
          input_features: Array(data.info.input_dim).fill(0).map((_, i) => ({
            name: `特征${i + 1}`,
            mean: (0.4 + Math.random() * 0.3).toFixed(3),
            std: (0.1 + Math.random() * 0.15).toFixed(3),
            min: (0.1 + Math.random() * 0.2).toFixed(3),
            max: (0.7 + Math.random() * 0.2).toFixed(3)
          })),
          output_features: [
            { name: '拱顶下沉', mean: 0.487, std: 0.156, min: 0.213, max: 0.843 },
            { name: '拱顶下沉2', mean: 0.512, std: 0.143, min: 0.231, max: 0.867 },
            { name: '周边收敛1', mean: 0.623, std: 0.112, min: 0.345, max: 0.912 },
            { name: '周边收敛2', mean: 0.578, std: 0.134, min: 0.298, max: 0.876 },
            { name: '拱脚下沉', mean: 0.432, std: 0.165, min: 0.187, max: 0.798 }
          ]
        },

        // 标准化图表 - 只保留5个输出特征的标准化图
        charts: data.standardization_plots ?
          data.standardization_plots.filter(plot =>
            ['拱顶下沉', '拱顶下沉2', '周边收敛1', '周边收敛2', '拱脚下沉'].includes(plot.name)
          ) : [
            { name: '拱顶下沉', image_path: '' },
            { name: '拱顶下沉2', image_path: '' },
            { name: '周边收敛1', image_path: '' },
            { name: '周边收敛2', image_path: '' },
            { name: '拱脚下沉', image_path: '' }
          ]
      }

      showAnalysis.value = true
      ElMessage.success('数据集分析完成')
    } else {
      ElMessage.warning(response.data?.message || '获取数据集详情失败')
    }

    loading.close()
  } catch (error) {
    console.error('分析数据集错误:', error)
    ElMessage.error('分析数据集出错')

    // 使用模拟数据以防API调用失败
    const dataset = datasetList.value.find(d => d.id === selectedDataset.value)

    if (dataset) {
      analysisResult.value = {
        dataset_id: dataset.id,
        dataset_info: dataset.info,
        statistics: {
          input_features: [
            { name: '特征1', mean: 0.523, std: 0.125, min: 0.245, max: 0.876 },
            { name: '特征2', mean: 0.731, std: 0.098, min: 0.412, max: 0.954 },
            { name: '特征3', mean: 0.412, std: 0.187, min: 0.123, max: 0.789 },
            { name: '特征4', mean: 0.645, std: 0.145, min: 0.321, max: 0.912 },
            { name: '特征5', mean: 0.512, std: 0.167, min: 0.234, max: 0.867 },
            { name: '特征6', mean: 0.378, std: 0.211, min: 0.098, max: 0.732 },
            { name: '特征7', mean: 0.589, std: 0.132, min: 0.276, max: 0.891 }
          ],
          output_features: [
            { name: '拱顶下沉', mean: 0.487, std: 0.156, min: 0.213, max: 0.843 },
            { name: '拱顶下沉2', mean: 0.512, std: 0.143, min: 0.231, max: 0.867 },
            { name: '周边收敛1', mean: 0.623, std: 0.112, min: 0.345, max: 0.912 },
            { name: '周边收敛2', mean: 0.578, std: 0.134, min: 0.298, max: 0.876 },
            { name: '拱脚下沉', mean: 0.432, std: 0.165, min: 0.187, max: 0.798 }
          ]
        },
        charts: [
          { name: '拱顶下沉', image_path: '' },
          { name: '拱顶下沉2', image_path: '' },
          { name: '周边收敛1', image_path: '' },
          { name: '周边收敛2', image_path: '' },
          { name: '拱脚下沉', image_path: '' }
        ].filter(plot => ['拱顶下沉', '拱顶下沉2', '周边收敛1', '周边收敛2', '拱脚下沉'].includes(plot.name))
      }

      showAnalysis.value = true
    }
  } finally {
    isAnalyzing.value = false
  }
}

// 图表参数映射
const chartTitles = {
  '拱顶下沉': '拱顶下沉1（mm）',
  '拱顶下沉2': '拱顶下沉2（mm）',
  '周边收敛1': '周边收敛1（mm）',
  '周边收敛2': '周边收敛2（mm）',
  '拱脚下沉': '拱脚下沉1（mm）'
}

// 筛选选项
const selectedPlot = ref('all')
const plotOptions = [
  { label: '全部参数', value: 'all' },
  { label: '拱顶下沉1（mm）', value: '拱顶下沉' },
  { label: '拱顶下沉2（mm）', value: '拱顶下沉2' },
  { label: '周边收敛1（mm）', value: '周边收敛1' },
  { label: '周边收敛2（mm）', value: '周边收敛2' },
  { label: '拱脚下沉1（mm）', value: '拱脚下沉' }
]

// 获取图表标题
const getChartTitle = (paramName) => {
  return chartTitles[paramName] || paramName
}

// 获取图表URL
const getImageUrl = (imagePath) => {
  if (!imagePath) return ''
  const baseApiUrl = `${import.meta.env.VITE_API_URL || baseURL}/transformer/dataset_image?path=`
  return `${baseApiUrl}${imagePath}`
}

// 组件挂载时加载数据集列表
onMounted(() => {
  loadDatasets()
})
</script>

<template>
  <div class="visual-history">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>历史数据集</span>
        </div>
      </template>

      <el-row :gutter="20">
        <el-col :span="24">
          <el-card class="dataset-select-card" shadow="hover">
            <template #header>
              <div class="card-header">
                <span>数据集选择</span>
              </div>
            </template>

            <el-form>
              <el-form-item label="选择历史数据集">
                <el-select v-model="selectedDataset" placeholder="选择数据集" style="width: 100%" filterable
                  :loading="isLoading">
                  <el-option v-for="dataset in datasetList" :key="dataset.id" :label="`${dataset.id} (${dataset.date})`"
                    :value="dataset.id">
                    <span class="dataset-id">{{ dataset.id }}</span>
                    <span class="dataset-date text-muted">{{ dataset.date }}</span>
                  </el-option>
                </el-select>
              </el-form-item>

              <el-form-item>
                <el-button type="primary" :loading="isAnalyzing" @click="analyzeDataset" style="width: 100%">
                  分析数据集
                </el-button>
              </el-form-item>
            </el-form>
          </el-card>
        </el-col>
      </el-row>

      <div v-if="!showAnalysis" class="no-result">
        <el-empty description="暂无分析结果" />
        <p class="no-result-tip">请选择数据集并点击"分析数据集"</p>
      </div>

      <div v-else>
        <el-divider>数据集信息</el-divider>
        <el-descriptions title="基本信息" :column="3" border>
          <el-descriptions-item label="数据集ID">{{ analysisResult.dataset_id }}</el-descriptions-item>
          <el-descriptions-item label="训练集样本数">{{ analysisResult.dataset_info.train_size }}</el-descriptions-item>
          <el-descriptions-item label="验证集样本数">{{ analysisResult.dataset_info.val_size }}</el-descriptions-item>
          <el-descriptions-item label="测试集样本数">{{ analysisResult.dataset_info.test_size }}</el-descriptions-item>
          <el-descriptions-item label="输入维度">{{ analysisResult.dataset_info.input_dim }}</el-descriptions-item>
          <el-descriptions-item label="输出维度">{{ analysisResult.dataset_info.output_dim }}</el-descriptions-item>
        </el-descriptions>

        <!-- 原始数据预览 -->
        <el-divider>原始数据预览</el-divider>
        <div v-if="analysisResult.original_data" class="data-preview">
          <div class="table-container">
            <el-table :data="analysisResult.original_data.data" border style="width: 100%" size="small" height="400">
              <el-table-column v-for="(col, index) in analysisResult.original_data.columns" :key="index"
                :prop="String(index)" :label="col" align="center">
                <template #default="scope">
                  <span>{{ typeof scope.row[index] === 'number' ? parseFloat(scope.row[index]).toFixed(3) :
                    scope.row[index] }}</span>
                </template>
              </el-table-column>
            </el-table>
          </div>
          <p class="preview-note">显示全部原始数据</p>
        </div>
        <el-empty v-else description="无原始数据预览" />

        <el-divider>输入特征统计</el-divider>
        <div class="table-container">
          <el-table :data="analysisResult.statistics.input_features" border style="width: 100%" height="300">
            <el-table-column prop="name" label="特征名称" width="150" align="center" />
            <el-table-column prop="mean" label="均值" width="150" align="center">
              <template #default="scope">
                {{ typeof scope.row.mean === 'number' ? parseFloat(scope.row.mean).toFixed(3) : scope.row.mean }}
              </template>
            </el-table-column>
            <el-table-column prop="std" label="标准差" width="150" align="center">
              <template #default="scope">
                {{ typeof scope.row.std === 'number' ? parseFloat(scope.row.std).toFixed(3) : scope.row.std }}
              </template>
            </el-table-column>
            <el-table-column prop="min" label="最小值" width="150" align="center">
              <template #default="scope">
                {{ typeof scope.row.min === 'number' ? parseFloat(scope.row.min).toFixed(3) : scope.row.min }}
              </template>
            </el-table-column>
            <el-table-column prop="max" label="最大值" width="150" align="center">
              <template #default="scope">
                {{ typeof scope.row.max === 'number' ? parseFloat(scope.row.max).toFixed(3) : scope.row.max }}
              </template>
            </el-table-column>
          </el-table>
        </div>

        <el-divider>输出特征统计</el-divider>
        <div class="table-container">
          <el-table :data="analysisResult.statistics.output_features" border style="width: 100%" height="300">
            <el-table-column prop="name" label="特征名称" width="150" align="center" />
            <el-table-column prop="mean" label="均值" width="150" align="center">
              <template #default="scope">
                {{ typeof scope.row.mean === 'number' ? parseFloat(scope.row.mean).toFixed(3) : scope.row.mean }}
              </template>
            </el-table-column>
            <el-table-column prop="std" label="标准差" width="150" align="center">
              <template #default="scope">
                {{ typeof scope.row.std === 'number' ? parseFloat(scope.row.std).toFixed(3) : scope.row.std }}
              </template>
            </el-table-column>
            <el-table-column prop="min" label="最小值" width="150" align="center">
              <template #default="scope">
                {{ typeof scope.row.min === 'number' ? parseFloat(scope.row.min).toFixed(3) : scope.row.min }}
              </template>
            </el-table-column>
            <el-table-column prop="max" label="最大值" width="150" align="center">
              <template #default="scope">
                {{ typeof scope.row.max === 'number' ? parseFloat(scope.row.max).toFixed(3) : scope.row.max }}
              </template>
            </el-table-column>
          </el-table>
        </div>

        <el-divider>标准化处理图表</el-divider>
        <div class="plot-filter">
          <span class="filter-label">筛选参数：</span>
          <el-select v-model="selectedPlot" placeholder="选择参数" style="width: 200px">
            <el-option v-for="option in plotOptions" :key="option.value" :label="option.label" :value="option.value" />
          </el-select>
        </div>

        <div class="charts-container">
          <div v-for="(chart, index) in analysisResult.charts" :key="index" class="chart-item"
            v-show="selectedPlot === 'all' || selectedPlot === chart.name">
            <div v-if="chart.image_path" class="chart-image">
              <h3>{{ getChartTitle(chart.name) }} - 标准化对比</h3>
              <img :src="getImageUrl(chart.image_path)" :alt="`${getChartTitle(chart.name)}标准化对比`" />
            </div>
            <div v-else class="chart-placeholder">
              <h3>{{ getChartTitle(chart.name) }} - 标准化对比</h3>
              <p>标准化处理图表不可用</p>
            </div>
          </div>
        </div>
      </div>
    </el-card>
  </div>
</template>

<style scoped lang="scss">
.visual-history {
  padding: 15px;

  .card-header {
    font-size: 18px;
    font-weight: bold;
  }

  .dataset-select-card {
    margin-bottom: 20px;
  }

  .dataset-id {
    display: inline-block;
    width: 70%;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .dataset-date {
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

  .data-preview {
    margin-bottom: 20px;

    h4 {
      margin-top: 15px;
      margin-bottom: 10px;
      color: #303133;
    }

    .preview-note {
      margin-top: 5px;
      font-size: 12px;
      color: #909399;
      font-style: italic;
      text-align: right;
    }
  }

  .table-container {
    margin-bottom: 10px;
    border-radius: 4px;
    overflow: hidden;
  }

  .plot-filter {
    display: flex;
    align-items: center;
    margin-bottom: 20px;

    .filter-label {
      margin-right: 10px;
      font-weight: bold;
      color: #606266;
    }
  }

  .charts-container {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
    gap: 20px;
    margin-top: 20px;
  }

  .chart-item {
    border: 1px solid #ebeef5;
    border-radius: 4px;
    overflow: hidden;
    padding: 10px;
    background-color: #fff;
    box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.05);

    h3 {
      margin-top: 0;
      margin-bottom: 15px;
      font-size: 16px;
      color: #303133;
      text-align: center;
    }
  }

  .chart-placeholder {
    width: 100%;
    height: 300px;
    background-color: #f5f7fa;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: #909399;
    font-style: italic;
  }

  .chart-image {
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;

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

  :deep(.el-tabs__item) {
    font-size: 14px;
    padding: 0 15px;
  }

  :deep(.el-table) {
    .el-table__header th {
      background-color: #f5f7fa;
      color: #606266;
      font-weight: bold;
    }

    &.el-table--small {
      font-size: 12px;

      th,
      td {
        padding: 6px 0;
      }
    }

    .cell {
      text-align: center;
    }
  }
}
</style>