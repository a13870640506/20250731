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

      // 确保标准化处理图表数据格式正确
      let standardizationPlots = [];
      if (data.standardization_plots && Array.isArray(data.standardization_plots)) {
        standardizationPlots = data.standardization_plots.map(plot => {
          // 确保每个图表对象包含必要的字段
          return {
            name: plot.name || '未命名参数',
            image: plot.image || null,  // 可能为空，将使用image_path
            image_path: plot.image_path || '',
            title: plot.title || plot.name || '标准化对比'
          };
        });

        console.log("处理后的标准化图表数据:", standardizationPlots);
      }

      // 构建分析结果
      analysisResult.value = {
        dataset_id: data.dataset_id,
        dataset_info: data.info,
        original_data: data.original_data,
        train_data: data.train_data,
        val_data: data.val_data,
        test_data: data.test_data,

        // 标准化处理图表
        standardization_plots: standardizationPlots
      }

      console.log("标准化处理图表数据:", standardizationPlots)

      showAnalysis.value = true
      ElMessage.success('数据集分析完成')
    } else {
      ElMessage.warning(response.data?.message || '获取数据集详情失败')
    }

    loading.close()
  } catch (error) {
    console.error('分析数据集错误:', error)
    ElMessage.error('分析数据集出错')
  } finally {
    isAnalyzing.value = false
  }
}

// 筛选选项
const selectedPlot = ref('all')

// 获取图表URL
const getImageUrl = (imagePath) => {
  if (!imagePath) return ''
  // 确保API基础URL正确
  const baseApiUrl = `${import.meta.env.VITE_API_URL || baseURL}/transformer/dataset_image?path=`
  console.log("构建图片URL:", baseApiUrl + encodeURIComponent(imagePath))
  return `${baseApiUrl}${encodeURIComponent(imagePath)}`
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
            <el-table :data="analysisResult.original_data.data" border style="width: 100%" size="small" height="500">
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



        <el-divider>标准化处理图表</el-divider>
        <div class="plot-filter-container">
          <div class="plot-filter">
            <span class="filter-label">筛选图表：</span>
            <el-select v-model="selectedPlot" placeholder="选择图表" style="width: 250px">
              <el-option label="全部图表" value="all" />
              <el-option v-for="(plot, index) in analysisResult.standardization_plots" :key="index" :label="plot.name"
                :value="plot.name" />
            </el-select>
          </div>
        </div>

        <div class="standardization-plots">
          <div v-if="analysisResult.standardization_plots && analysisResult.standardization_plots.length > 0">
            <div v-for="(plot, index) in analysisResult.standardization_plots" :key="index" class="plot-item"
              v-show="selectedPlot === 'all' || selectedPlot === plot.name">
              <div class="plot-image-container">
                <img v-if="plot.image" :src="`data:image/png;base64,${plot.image}`" :alt="plot.name"
                  class="plot-image" />
                <img v-else-if="plot.image_path" :src="getImageUrl(plot.image_path)" :alt="plot.name"
                  class="plot-image" />
                <div v-else class="placeholder-charts">
                  <p class="chart-placeholder">该参数没有可用的标准化处理图</p>
                </div>
              </div>
              <div class="plot-footer">
                <a v-if="plot.image_path"
                  :href="`http://127.0.0.1:8000/download?path=${encodeURIComponent(plot.image_path)}`" target="_blank"
                  class="download-link">
                  下载高清图
                </a>
              </div>
            </div>
          </div>
          <el-empty v-else description="暂无标准化处理图表" />
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

  .plot-filter-container {
    display: flex;
    justify-content: center;
    margin-bottom: 20px;
  }

  .plot-filter {
    display: flex;
    align-items: center;

    .filter-label {
      margin-right: 10px;
      font-weight: bold;
      color: #606266;
    }
  }

  .standardization-plots {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 30px;
    margin-top: 20px;

    .plot-item {
      width: 90%;
      max-width: 800px;
      text-align: center;

      .plot-image-container {
        padding: 5px;
        background-color: white;
        border-radius: 4px;
      }

      .plot-image {
        width: 100%;
        border-radius: 4px;
      }

      .plot-footer {
        margin-top: 10px;
        display: flex;
        justify-content: flex-end;
        align-items: center;

        .download-link {
          color: #409EFF;
          font-size: 13px;
          text-decoration: none;

          &:hover {
            text-decoration: underline;
          }
        }
      }
    }
  }

  .placeholder-charts {
    margin-top: 20px;
    height: 300px;
    background-color: #f5f7fa;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;

    .chart-placeholder {
      color: #909399;
      font-style: italic;
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