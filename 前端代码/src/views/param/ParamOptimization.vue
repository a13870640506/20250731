<script setup>
import { ref, reactive, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { optimizeParamsService, getOptimizationHistoryService, getOptimizationResultService, getDatasetsService } from '@/api/transformer'
import { useRoute } from 'vue-router'
import { Plus, Download, Document } from '@element-plus/icons-vue'

// 获取路由参数
const route = useRoute()

// 创建响应式数据存储用户输入的内容
const formData = ref({
  n_trials: 50,
  epochs_per_trial: 30,
  batch_size: 16,
  data_path: route.query.data_path || '', // 从URL查询参数获取数据集路径
  // 参照main.py中的objective函数定义参数范围
  d_model_values: [64, 128, 256], // 离散值
  nhead_values: [1, 2, 4], // 离散值
  num_layers_min: 1, // 连续值范围
  num_layers_max: 4,
  lr_min: 1e-5, // 连续值范围（对数均匀分布）
  lr_max: 1e-3,
  weight_decay_min: 1e-8, // 连续值范围（对数均匀分布）
  weight_decay_max: 1e-4
})

// 优化状态
const isOptimizing = ref(false)

// 优化结果
const optimizationResult = ref({
  best_params: null,
  metrics: null,
  optimization_id: null,
  curve_image: null,
  curve_path: null,
  csv_path: null,
  trials: []
})

// 优化历史记录
const optimizationHistory = ref([])

// 是否显示结果
const showResult = ref(false)

// 历史记录对话框
const historyDialogVisible = ref(false)

// 当前选中的Tab
const activeTab = ref('optimize')

// 数据集列表
const datasets = ref([])

// 可选值列表
const availableValues = ref({
  d_model: [64, 128, 256],
  nhead: [1, 2, 4]
})

// 添加值对话框
const addValueDialogVisible = ref(false)
const currentParamType = ref('')
const newParamValue = ref(null)

// 提交表单开始优化
const submitForm = async () => {
  try {
    // 验证是否选择了数据集
    if (!formData.value.data_path) {
      ElMessage.warning('请选择数据集')
      return
    }

    isOptimizing.value = true

    // 创建FormData对象
    const formDataToUpload = new FormData()
    formDataToUpload.append('n_trials', formData.value.n_trials)
    formDataToUpload.append('epochs_per_trial', formData.value.epochs_per_trial)
    formDataToUpload.append('batch_size', formData.value.batch_size)
    formDataToUpload.append('data_path', formData.value.data_path)

    // 添加参数范围
    formDataToUpload.append('d_model_values', JSON.stringify(formData.value.d_model_values))
    formDataToUpload.append('nhead_values', JSON.stringify(formData.value.nhead_values))
    formDataToUpload.append('num_layers_min', formData.value.num_layers_min)
    formDataToUpload.append('num_layers_max', formData.value.num_layers_max)
    formDataToUpload.append('lr_min', formData.value.lr_min)
    formDataToUpload.append('lr_max', formData.value.lr_max)
    formDataToUpload.append('weight_decay_min', formData.value.weight_decay_min)
    formDataToUpload.append('weight_decay_max', formData.value.weight_decay_max)

    ElMessage.info('超参数优化中，请稍候...')

    // 调用API
    const res = await optimizeParamsService(formDataToUpload)

    if (res.data && res.data.success) {
      ElMessage.success('超参数优化成功')
      optimizationResult.value = res.data.data
      showResult.value = true
      activeTab.value = 'result'

      // 刷新优化历史记录
      loadOptimizationHistory()
    } else {
      ElMessage.error(res.data?.message || '超参数优化失败')
    }
    isOptimizing.value = false
  } catch (error) {
    ElMessage.error(`优化失败: ${error.message}`)
    console.error('优化错误:', error)
    isOptimizing.value = false
  }
}

// 加载优化历史记录
const loadOptimizationHistory = async () => {
  try {
    // 调用API获取优化历史记录
    const res = await getOptimizationHistoryService()

    if (res.data && res.data.success) {
      optimizationHistory.value = res.data.data
    }
  } catch (error) {
    console.error('加载优化历史记录失败:', error)
    ElMessage.error('加载优化历史记录失败')
  }
}

// 获取优化结果详情
const getOptimizationDetail = async (optId) => {
  try {
    const res = await getOptimizationResultService(optId)

    if (res.data && res.data.success) {
      return res.data.data
    } else {
      ElMessage.error(res.data?.message || '获取优化详情失败')
      return null
    }
  } catch (error) {
    console.error('获取优化详情失败:', error)
    ElMessage.error('获取优化详情失败')
    return null
  }
}

// 使用历史优化结果
const useHistoryResult = async (result) => {
  // 获取完整的优化结果详情
  const detail = await getOptimizationDetail(result.id)

  if (detail) {
    optimizationResult.value = {
      best_params: detail.best_params,
      metrics: detail.metrics,
      optimization_id: detail.id,
      curve_image: detail.curve_image,
      curve_path: detail.curve_path,
      csv_path: detail.csv_path,
      trials: detail.trials
    }
    showResult.value = true
    activeTab.value = 'result'
    historyDialogVisible.value = false
  }
}

// 打开历史记录对话框
const openHistoryDialog = () => {
  loadOptimizationHistory()
  historyDialogVisible.value = true
}

// 下载文件
const downloadFile = (path, type) => {
  if (path) {
    window.open(`http://127.0.0.1:8000/download?path=${encodeURIComponent(path)}`, '_blank')
  } else {
    ElMessage.warning(`找不到${type === 'csv' ? 'CSV文件' : '图像文件'}路径`)
  }
}

// 下载CSV文件
const downloadCSV = (path) => {
  downloadFile(path, 'csv')
}

// 下载高清图
const downloadImage = (path) => {
  downloadFile(path, 'image')
}

// 表格行的样式
const tableRowClassName = ({ row }) => {
  if (row.number === optimizationResult.value.metrics?.best_trial) {
    return 'best-row'
  }
  return ''
}

// 格式化数值
const formatNumber = (value, precision = 6) => {
  if (value === undefined || value === null) return '-'
  return Number(value).toFixed(precision)
}

// 格式化科学计数法
const formatExponential = (value, precision = 6) => {
  if (value === undefined || value === null) return '-'
  return Number(value).toExponential(precision)
}

// 组件挂载时初始化
// 加载数据集列表
const loadDatasets = async () => {
  try {
    const res = await getDatasetsService()
    if (res.data && res.data.success) {
      datasets.value = res.data.data || []
    } else {
      ElMessage.error(res.data?.message || '获取数据集列表失败')
    }
  } catch (error) {
    console.error('获取数据集列表错误:', error)
    ElMessage.error(`获取数据集列表失败: ${error.message}`)
  }
}

// 显示添加值对话框
const showAddValueDialog = (paramType) => {
  currentParamType.value = paramType
  newParamValue.value = null
  addValueDialogVisible.value = true
}

// 添加新值
const addNewValue = () => {
  if (!newParamValue.value) {
    ElMessage.warning('请输入有效的参数值')
    return
  }

  const value = Number(newParamValue.value)
  if (isNaN(value)) {
    ElMessage.warning('请输入有效的数字')
    return
  }

  // 检查是否已存在
  if (availableValues.value[currentParamType.value].includes(value)) {
    ElMessage.warning('该值已存在')
    return
  }

  // 添加新值
  availableValues.value[currentParamType.value].push(value)
  // 排序
  availableValues.value[currentParamType.value].sort((a, b) => a - b)

  // 如果当前选择中没有值，自动选中新添加的值
  if (formData.value[`${currentParamType.value}_values`].length === 0) {
    formData.value[`${currentParamType.value}_values`].push(value)
  }

  addValueDialogVisible.value = false
  ElMessage.success('添加成功')
}

onMounted(async () => {
  // 加载优化历史记录
  await loadOptimizationHistory()
  // 加载数据集列表
  await loadDatasets()

  // 检查URL中是否有优化ID参数，如果有则自动加载对应的优化结果
  const optId = route.query.id
  if (optId) {
    const detail = await getOptimizationDetail(optId)
    if (detail) {
      optimizationResult.value = {
        best_params: detail.best_params,
        metrics: detail.metrics,
        optimization_id: detail.id,
        curve_image: detail.curve_image,
        curve_path: detail.curve_path,
        csv_path: detail.csv_path,
        trials: detail.trials
      }
      showResult.value = true
      activeTab.value = 'result'
    }
  }
})
</script>

<template>
  <div class="param-optimization">
    <div class="page-header">
      <div class="title">超参数优化</div>
      <div class="actions">
        <el-button type="primary" @click="openHistoryDialog" size="small">历史优化记录</el-button>
        <el-button type="default" @click="activeTab = 'optimize'" size="small" v-if="showResult">返回设置</el-button>
      </div>
    </div>

    <!-- 优化设置Tab -->
    <div v-if="activeTab === 'optimize'" class="optimization-settings">
      <el-card shadow="never">
        <div class="settings-header">超参数优化设置</div>

        <el-form :model="formData" label-position="top" label-width="120px">
          <el-row :gutter="20">
            <!-- 优化参数 -->
            <el-col :span="12">
              <el-form-item label="选择数据集" required>
                <el-select v-model="formData.data_path" placeholder="请选择数据集" style="width: 100%">
                  <el-option v-for="dataset in datasets" :key="dataset.path"
                    :label="`${dataset.date} (训练集: ${dataset.info['训练集样本数'] || '?'}, 验证集: ${dataset.info['验证集样本数'] || '?'})`"
                    :value="dataset.path" />
                </el-select>
              </el-form-item>

              <el-form-item label="优化轮次">
                <el-input-number v-model="formData.n_trials" :min="10" :max="200" style="width: 100%" />
              </el-form-item>

              <el-form-item label="每轮训练轮次">
                <el-input-number v-model="formData.epochs_per_trial" :min="5" :max="100" style="width: 100%" />
              </el-form-item>

              <el-form-item label="批次大小">
                <el-input-number v-model="formData.batch_size" :min="1" :max="128" style="width: 100%" />
              </el-form-item>
            </el-col>

            <!-- 离散参数 -->
            <el-col :span="12">
              <el-form-item label="模型维度 (d_model)">
                <div class="discrete-param-container">
                  <el-select v-model="formData.d_model_values" multiple placeholder="选择可能的值"
                    style="width: calc(100% - 40px)">
                    <el-option v-for="value in availableValues.d_model" :key="value" :value="value"
                      :label="value.toString()" />
                  </el-select>
                  <el-button type="primary" @click="showAddValueDialog('d_model')" class="add-value-btn">
                    <el-icon>
                      <Plus />
                    </el-icon>
                  </el-button>
                </div>
              </el-form-item>

              <el-form-item label="注意力头数 (nhead)">
                <div class="discrete-param-container">
                  <el-select v-model="formData.nhead_values" multiple placeholder="选择可能的值"
                    style="width: calc(100% - 40px)">
                    <el-option v-for="value in availableValues.nhead" :key="value" :value="value"
                      :label="value.toString()" />
                  </el-select>
                  <el-button type="primary" @click="showAddValueDialog('nhead')" class="add-value-btn">
                    <el-icon>
                      <Plus />
                    </el-icon>
                  </el-button>
                </div>
              </el-form-item>
            </el-col>
          </el-row>

          <el-divider>连续参数范围</el-divider>

          <el-row :gutter="20">
            <el-col :span="8">
              <el-form-item label="Transformer层数范围">
                <el-row :gutter="10">
                  <el-col :span="12">
                    <el-input-number v-model="formData.num_layers_min" :min="1" :max="4" placeholder="最小值"
                      style="width: 100%" />
                  </el-col>
                  <el-col :span="12">
                    <el-input-number v-model="formData.num_layers_max" :min="1" :max="4" placeholder="最大值"
                      style="width: 100%" />
                  </el-col>
                </el-row>
              </el-form-item>
            </el-col>

            <el-col :span="8">
              <el-form-item label="学习率范围">
                <el-row :gutter="10">
                  <el-col :span="12">
                    <el-input v-model="formData.lr_min" placeholder="最小值" />
                  </el-col>
                  <el-col :span="12">
                    <el-input v-model="formData.lr_max" placeholder="最大值" />
                  </el-col>
                </el-row>
              </el-form-item>
            </el-col>

            <el-col :span="8">
              <el-form-item label="权重衰减范围">
                <el-row :gutter="10">
                  <el-col :span="12">
                    <el-input v-model="formData.weight_decay_min" placeholder="最小值" />
                  </el-col>
                  <el-col :span="12">
                    <el-input v-model="formData.weight_decay_max" placeholder="最大值" />
                  </el-col>
                </el-row>
              </el-form-item>
            </el-col>
          </el-row>

          <div class="action-buttons">
            <el-button type="primary" :loading="isOptimizing" @click="submitForm">开始优化</el-button>
          </div>
        </el-form>
      </el-card>
    </div>

    <!-- 优化结果Tab -->
    <div v-if="activeTab === 'result' && showResult" class="optimization-result">
      <!-- 优化结果页面标签页 -->
      <el-tabs type="card" class="result-tabs">
        <!-- 优化曲线标签页 -->
        <el-tab-pane label="优化曲线">
          <el-row :gutter="20">
            <!-- 优化曲线 -->
            <el-col :span="16">
              <el-card shadow="hover" class="chart-card">
                <template #header>
                  <div class="card-header">
                    <span>优化过程曲线</span>
                  </div>
                </template>
                <div v-if="optimizationResult.curve_image" class="curve-container">
                  <img :src="`data:image/png;base64,${optimizationResult.curve_image}`" alt="优化过程曲线"
                    class="curve-image" />
                  <div class="curve-actions">
                    <el-button type="primary" size="small" @click="downloadImage(optimizationResult.curve_path)">
                      <el-icon>
                        <Download />
                      </el-icon> 下载高清图
                    </el-button>
                    <el-button type="success" size="small" @click="downloadCSV(optimizationResult.csv_path)">
                      <el-icon>
                        <Document />
                      </el-icon> 下载CSV数据
                    </el-button>
                  </div>
                </div>
              </el-card>
            </el-col>

            <!-- 最佳参数 -->
            <el-col :span="8">
              <el-card shadow="hover" class="params-card">
                <template #header>
                  <div class="card-header">
                    <span>最佳参数</span>
                  </div>
                </template>
                <div class="params-list" v-if="optimizationResult.best_params">
                  <div class="param-item">
                    <span class="param-name">d_model:</span>
                    <span class="param-value">{{ optimizationResult.best_params.d_model }}</span>
                  </div>
                  <div class="param-item">
                    <span class="param-name">nhead:</span>
                    <span class="param-value">{{ optimizationResult.best_params.nhead }}</span>
                  </div>
                  <div class="param-item">
                    <span class="param-name">num_layers:</span>
                    <span class="param-value">{{ optimizationResult.best_params.num_layers }}</span>
                  </div>
                  <div class="param-item">
                    <span class="param-name">batch_size:</span>
                    <span class="param-value">{{ optimizationResult.best_params.batch_size }}</span>
                  </div>
                  <div class="param-item">
                    <span class="param-name">lr:</span>
                    <span class="param-value highlight">{{ formatExponential(optimizationResult.best_params.lr)
                    }}</span>
                  </div>
                  <div class="param-item">
                    <span class="param-name">weight_decay:</span>
                    <span class="param-value">{{ formatExponential(optimizationResult.best_params.weight_decay)
                    }}</span>
                  </div>
                </div>

                <div class="metrics-section">
                  <div class="metrics-title">优化指标</div>
                  <div class="metric-item" v-if="optimizationResult.metrics">
                    <span class="metric-name">验证损失 (MSE):</span>
                    <span class="metric-value">{{ formatNumber(optimizationResult.metrics.val_loss) }}</span>
                  </div>
                  <div class="metric-item" v-if="optimizationResult.metrics">
                    <span class="metric-name">优化轮次:</span>
                    <span class="metric-value">{{ optimizationResult.metrics.best_trial + 1 }}</span>
                  </div>
                </div>

                <!-- 已移动到曲线下方 -->
              </el-card>
            </el-col>
          </el-row>
        </el-tab-pane>

        <!-- 优化表格标签页 -->
        <el-tab-pane label="优化表格">
          <el-card shadow="hover" class="trials-card">
            <template #header>
              <div class="card-header">
                <span>优化试验记录</span>
              </div>
            </template>

            <el-table v-if="optimizationResult.trials && optimizationResult.trials.length > 0"
              :data="optimizationResult.trials" style="width: 100%" height="500" stripe border
              :row-class-name="tableRowClassName">
              <el-table-column label="轮次" width="70" align="center">
                <template #default="scope">
                  {{ scope.row.number + 1 }}
                </template>
              </el-table-column>
              <el-table-column prop="value" label="验证损失" width="120" align="center">
                <template #default="scope">
                  <span>
                    {{ formatNumber(scope.row.value) }}
                  </span>
                </template>
              </el-table-column>
              <el-table-column label="d_model" width="100" align="center">
                <template #default="scope">
                  {{ scope.row.d_model || '-' }}
                </template>
              </el-table-column>
              <el-table-column label="nhead" width="100" align="center">
                <template #default="scope">
                  {{ scope.row.nhead || '-' }}
                </template>
              </el-table-column>
              <el-table-column label="num_layers" width="100" align="center">
                <template #default="scope">
                  {{ scope.row.num_layers || '-' }}
                </template>
              </el-table-column>
              <el-table-column label="lr" width="120" align="center">
                <template #default="scope">
                  {{ scope.row.lr !== undefined ? formatExponential(scope.row.lr) : '-' }}
                </template>
              </el-table-column>
              <el-table-column label="weight_decay" width="120" align="center">
                <template #default="scope">
                  {{ scope.row.weight_decay !== undefined ? formatExponential(scope.row.weight_decay) : '-' }}
                </template>
              </el-table-column>
              <!-- batch_size列已删除 -->
              <el-table-column label="dropout" width="100" align="center">
                <template #default="scope">
                  {{ scope.row.dropout !== undefined ? scope.row.dropout : '-' }}
                </template>
              </el-table-column>
              <el-table-column label="最佳" width="80" align="center">
                <template #default="scope">
                  <el-tag type="success" effect="dark" size="small"
                    v-if="scope.row.number === optimizationResult.metrics.best_trial">最佳</el-tag>
                </template>
              </el-table-column>
            </el-table>

            <el-empty v-else description="暂无数据" />
          </el-card>
        </el-tab-pane>
      </el-tabs>
    </div>

    <!-- 历史优化记录对话框 -->
    <el-dialog v-model="historyDialogVisible" title="历史优化记录" width="80%" destroy-on-close>
      <el-table :data="optimizationHistory" style="width: 100%" v-if="optimizationHistory.length > 0" stripe border>
        <el-table-column prop="id" label="优化ID" width="180" />
        <el-table-column prop="date" label="日期" width="180" />
        <el-table-column label="最优参数" width="300">
          <template #default="scope">
            <div>d_model: {{ scope.row.best_params.d_model }}</div>
            <div>nhead: {{ scope.row.best_params.nhead }}</div>
            <div>num_layers: {{ scope.row.best_params.num_layers }}</div>
            <div>lr: {{ formatExponential(scope.row.best_params.lr) }}</div>
            <div>weight_decay: {{ formatExponential(scope.row.best_params.weight_decay) }}</div>
          </template>
        </el-table-column>
        <el-table-column label="指标">
          <template #default="scope">
            <div>验证损失: {{ formatNumber(scope.row.metrics?.val_loss) }}</div>
            <div>最优轮次: {{ scope.row.metrics?.best_trial !== undefined ? scope.row.metrics.best_trial + 1 : 'N/A' }}
            </div>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="200">
          <template #default="scope">
            <el-button type="primary" size="small" @click="useHistoryResult(scope.row)">
              查看详情
            </el-button>
            <el-button type="success" size="small" v-if="scope.row.csv_path" @click="downloadCSV(scope.row.csv_path)">
              下载数据
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <div v-else class="no-history">
        <el-empty description="暂无优化历史记录" />
      </div>
    </el-dialog>

    <!-- 添加参数值对话框 -->
    <el-dialog v-model="addValueDialogVisible" :title="`添加${currentParamType === 'd_model' ? '模型维度' : '注意力头数'}值`"
      width="30%">
      <el-form>
        <el-form-item :label="currentParamType === 'd_model' ? '模型维度值' : '注意力头数值'">
          <el-input-number v-model="newParamValue" :min="1" :step="currentParamType === 'd_model' ? 64 : 1"
            style="width: 100%" />
        </el-form-item>
      </el-form>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="addValueDialogVisible = false">取消</el-button>
          <el-button type="primary" @click="addNewValue">确认</el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped lang="scss">
.param-optimization {
  padding: 20px;
  font-family: "Microsoft YaHei", "微软雅黑", sans-serif;

  .discrete-param-container {
    display: flex;
    align-items: center;
    gap: 10px;

    .add-value-btn {
      padding: 8px;
      height: 32px;
    }
  }

  .page-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;

    .title {
      font-size: 18px;
      font-weight: bold;
    }

    .actions {
      display: flex;
      gap: 10px;
    }
  }

  .result-tabs {
    .el-tabs__header {
      margin-bottom: 20px;
    }

    .el-tabs__item {
      font-size: 15px;
      height: 40px;
      line-height: 40px;
    }
  }

  .card-header {
    font-size: 16px;
    font-weight: bold;
  }

  .settings-header,
  .params-title,
  .metrics-title,
  .trials-title,
  .chart-title {
    font-size: 16px;
    font-weight: bold;
    margin-bottom: 15px;
    color: #409EFF;
    border-bottom: 1px solid #EBEEF5;
    padding-bottom: 8px;
  }

  .action-buttons {
    display: flex;
    justify-content: center;
    margin-top: 20px;
  }

  .chart-card,
  .params-card,
  .trials-card {
    margin-bottom: 20px;
    box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  }

  .curve-container {
    text-align: center;

    .curve-image {
      max-width: 100%;
      border-radius: 4px;
      margin-bottom: 15px;
    }

    .curve-actions {
      display: flex;
      justify-content: center;
      gap: 15px;
      margin-top: 10px;
      margin-bottom: 5px;
    }
  }

  .params-list {
    margin-bottom: 20px;

    .param-item {
      display: flex;
      justify-content: space-between;
      margin-bottom: 10px;
      padding: 8px;
      background-color: #f8f9fa;
      border-radius: 4px;

      .param-name {
        color: #606266;
        font-family: "Microsoft YaHei", "微软雅黑", sans-serif;
      }

      .param-value {
        font-family: "Microsoft YaHei", "微软雅黑", sans-serif;
        font-weight: 500;

        &.highlight {
          color: #409EFF;
          font-weight: bold;
        }
      }
    }
  }

  .metrics-section {
    margin-top: 20px;

    .metrics-title {
      font-family: "Microsoft YaHei", "微软雅黑", sans-serif;
    }

    .metric-item {
      display: flex;
      justify-content: space-between;
      margin-bottom: 10px;
      padding: 8px;
      background-color: #f8f9fa;
      border-radius: 4px;

      .metric-name {
        color: #606266;
        font-family: "Microsoft YaHei", "微软雅黑", sans-serif;
      }

      .metric-value {
        font-family: "Microsoft YaHei", "微软雅黑", sans-serif;
        font-weight: 500;
        color: #67C23A;
      }
    }
  }

  .best-value {
    color: #67C23A;
    font-weight: bold;
  }

  // 最优行的样式
  :deep(.best-row) {
    color: #67C23A;
    font-weight: bold;
  }

  // 表格内容居中显示
  :deep(.el-table .cell) {
    text-align: center;
  }

  .no-history {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 200px;
  }
}
</style>