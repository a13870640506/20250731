<script setup>
import { ref, reactive, onMounted } from 'vue'
import { ElMessage, ElLoading } from 'element-plus'

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

    // 模拟API调用
    setTimeout(() => {
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
        },
        {
          id: 'processed_20250801_103814',
          date: '2025-08-01 10:38:14',
          info: {
            train_size: 180,
            val_size: 30,
            test_size: 60,
            input_dim: 7,
            output_dim: 5
          }
        }
      ]
      isLoading.value = false
    }, 500)
  } catch (error) {
    console.error('获取历史数据集列表错误:', error)
    ElMessage.error('获取历史数据集列表出错')
    isLoading.value = false
  }
}

// 分析数据集
const analyzeDataset = () => {
  if (!selectedDataset.value) {
    ElMessage.warning('请先选择数据集')
    return
  }

  isAnalyzing.value = true

  // 模拟API调用
  setTimeout(() => {
    // 查找选中的数据集
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
        ]
      }
    }

    showAnalysis.value = true
    isAnalyzing.value = false
    ElMessage.success('数据集分析完成')
  }, 1500)
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

        <el-divider>输入特征统计</el-divider>
        <el-table :data="analysisResult.statistics.input_features" border style="width: 100%">
          <el-table-column prop="name" label="特征名称" width="150" />
          <el-table-column prop="mean" label="均值" width="150" />
          <el-table-column prop="std" label="标准差" width="150" />
          <el-table-column prop="min" label="最小值" width="150" />
          <el-table-column prop="max" label="最大值" width="150" />
        </el-table>

        <el-divider>输出特征统计</el-divider>
        <el-table :data="analysisResult.statistics.output_features" border style="width: 100%">
          <el-table-column prop="name" label="特征名称" width="150" />
          <el-table-column prop="mean" label="均值" width="150" />
          <el-table-column prop="std" label="标准差" width="150" />
          <el-table-column prop="min" label="最小值" width="150" />
          <el-table-column prop="max" label="最大值" width="150" />
        </el-table>

        <el-divider>标准化处理图表</el-divider>
        <div class="charts-container">
          <div v-for="(chart, index) in analysisResult.charts" :key="index" class="chart-item">
            <div class="chart-placeholder">
              <h3>{{ chart.name }} - 标准化对比</h3>
              <p>标准化处理图表将显示在这里</p>
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

  .text-muted {
    color: #909399;
  }
}
</style>