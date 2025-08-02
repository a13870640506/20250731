<script setup>
import { ref, reactive, onMounted } from 'vue'
import { ElMessage, ElDialog, ElMessageBox } from 'element-plus'
import { InfoFilled, Timer, Clock, Document, Delete, View } from '@element-plus/icons-vue'
import {
  predictService,
  getTunnelModelsService,
  getPredictionHistoryService,
  deletePredictionHistoryService,
  getPredictionDetailService
} from '@/api/transformer'

// 创建响应式数据存储用户输入的围岩参数
const inputParams = reactive({
  poissonRatio: 0.39,        // 泊松比
  frictionAngle: 26.94,        // 内摩擦角（°）
  cohesion: 0.475,            // 粘聚力（Mpa）
  dilationAngle: 8.47,         // 剪胀角（°）
  elasticModulus: 16.13      // 弹性模量（MPa）
})

// 预测结果
const predictionResult = reactive({
  crownSettlement1: null,   // 拱顶下沉1（mm）
  crownSettlement2: null,   // 拱顶下沉2（mm）
  convergence1: null,       // 周边收敛1（mm）
  convergence2: null,       // 周边收敛2（mm）
  footSettlement: null,     // 拱脚下沉1（mm）
  predictionTime: null,     // 预测耗时（ms）
  metrics: {
    r2: null,
    mse: null,
    mape: null
  }
})

// 模型相关数据
const modelData = reactive({
  modelPath: '',
  availableModels: []
})

// 历史预测记录
const predictionHistory = ref([])
const historyDialogVisible = ref(false)
const loadingHistory = ref(false)

// 加载状态
const isLoading = ref(false)
const showResult = ref(false)

// 从后端获取可用模型列表
const fetchAvailableModels = async () => {
  try {
    isLoading.value = true
    const res = await getTunnelModelsService()
    if (res.data && res.data.success) {
      modelData.availableModels = res.data.data || []
      if (modelData.availableModels.length > 0) {
        // 默认选择第一个模型
        modelData.modelPath = modelData.availableModels[0].path
      }
    } else {
      ElMessage.warning('获取模型列表失败')
    }
  } catch (error) {
    console.error('获取模型列表错误:', error)
    ElMessage.error(`获取模型列表失败: ${error.message}`)
  } finally {
    isLoading.value = false
  }
}

// 执行预测
const predict = async () => {
  if (!modelData.modelPath) {
    ElMessage.warning('请选择模型路径')
    return
  }

  try {
    isLoading.value = true
    ElMessage.info('正在进行预测，请稍候...')

    console.log('预测请求参数:', {
      modelPath: modelData.modelPath,
      poissonRatio: inputParams.poissonRatio,
      frictionAngle: inputParams.frictionAngle,
      cohesion: inputParams.cohesion,
      dilationAngle: inputParams.dilationAngle,
      elasticModulus: inputParams.elasticModulus
    })

    // 每次新预测前先清空旧结果
    showResult.value = false

    // 准备请求数据（确保为数字类型）
    const requestData = {
      model_path: modelData.modelPath,
      input_params: {
        poisson_ratio: Number(inputParams.poissonRatio),
        friction_angle: Number(inputParams.frictionAngle),
        cohesion: Number(inputParams.cohesion),
        dilation_angle: Number(inputParams.dilationAngle),
        elastic_modulus: Number(inputParams.elasticModulus)
      }
    }

    // 发送预测请求
    const res = await predictService(requestData)
    console.log('预测响应:', res)

    if (res.data && res.data.success) {
      // 提取预测时间
      const predictionTimeMs = res.data.data.prediction_time_ms || 0

      // 更新预测结果
      const data = res.data.data
      predictionResult.crownSettlement1 = data.crown_settlement1
      predictionResult.crownSettlement2 = data.crown_settlement2
      predictionResult.convergence1 = data.convergence1
      predictionResult.convergence2 = data.convergence2
      predictionResult.footSettlement = data.foot_settlement
      predictionResult.predictionTime = predictionTimeMs

      // 更新评估指标
      predictionResult.metrics = data.metrics || {
        r2: null,
        mse: null,
        mape: null
      }

      // 显示结果
      showResult.value = true

      // 显示成功消息，包含预测时间
      ElMessage.success(`预测成功，耗时${predictionTimeMs}ms`)

      // 刷新历史记录
      fetchPredictionHistory()
    } else {
      ElMessage.error(res.data?.message || '预测失败')
    }
  } catch (error) {
    ElMessage.error(`预测失败: ${error.message}`)
    console.error('预测错误:', error)
  } finally {
    isLoading.value = false
  }
}

// 获取历史预测记录
const fetchPredictionHistory = async () => {
  try {
    loadingHistory.value = true
    const res = await getPredictionHistoryService()
    if (res.data && res.data.success) {
      predictionHistory.value = res.data.data || []
      console.log(`成功获取${predictionHistory.value.length}条历史预测记录`)
    } else {
      console.warn('获取历史预测记录失败:', res.data?.message)
      ElMessage.warning('获取历史预测记录失败，请检查后端服务是否正常运行')
    }
  } catch (error) {
    console.error('获取历史预测记录错误:', error)
    ElMessage.error(`获取历史预测记录失败: ${error.message}`)
  } finally {
    loadingHistory.value = false
  }
}

// 打开历史预测记录对话框
const openHistoryDialog = () => {
  fetchPredictionHistory()
  historyDialogVisible.value = true
}

// 查看历史记录详情
const viewHistoryDetail = async (record) => {
  try {
    if (!record || !record.id) {
      ElMessage.warning('记录ID无效')
      return
    }

    ElMessage.info('正在加载预测记录详情...')

    // 获取预测记录详情
    const res = await getPredictionDetailService(record.id)

    if (res.data && res.data.success) {
      // 显示详情数据
      const detailData = res.data.data

      // 填充输入参数
      if (detailData.input_params) {
        inputParams.poissonRatio = detailData.input_params.poisson_ratio
        inputParams.frictionAngle = detailData.input_params.friction_angle
        inputParams.cohesion = detailData.input_params.cohesion
        inputParams.dilationAngle = detailData.input_params.dilation_angle
        inputParams.elasticModulus = detailData.input_params.elastic_modulus
      }

      // 填充预测结果
      if (detailData.prediction_results) {
        predictionResult.crownSettlement1 = detailData.prediction_results.crown_settlement1
        predictionResult.crownSettlement2 = detailData.prediction_results.crown_settlement2
        predictionResult.convergence1 = detailData.prediction_results.convergence1
        predictionResult.convergence2 = detailData.prediction_results.convergence2
        predictionResult.footSettlement = detailData.prediction_results.foot_settlement
        predictionResult.predictionTime = detailData.prediction_time_ms
      }

      // 如果模型路径存在于可用模型列表中，则选择该模型
      if (detailData.model_path && modelData.availableModels.some(model => model.path === detailData.model_path)) {
        modelData.modelPath = detailData.model_path
      }

      // 显示结果
      showResult.value = true

      // 关闭对话框
      historyDialogVisible.value = false

      // 提示用户
      ElMessage.success('已加载历史预测记录')
    } else {
      ElMessage.error(res.data?.message || '获取预测记录详情失败')
    }
  } catch (error) {
    ElMessage.error(`获取预测记录详情失败: ${error.message}`)
    console.error('获取预测记录详情错误:', error)
  }
}

// 删除历史记录
const deleteHistoryRecord = async (record) => {
  try {
    if (!record || !record.id) {
      ElMessage.warning('记录ID无效')
      return
    }

    // 使用Element Plus确认对话框
    try {
      await ElMessageBox.confirm(
        `确定要删除ID为${record.id}的预测记录吗？此操作将同时删除服务器上存储的预测结果文件，且不可恢复。`,
        '删除确认',
        {
          confirmButtonText: '确定删除',
          cancelButtonText: '取消',
          type: 'warning',
          distinguishCancelAndClose: true,
          closeOnClickModal: false
        }
      )

      // 用户点击了确认，执行删除操作
      ElMessage.info('正在删除预测记录...')
      const res = await deletePredictionHistoryService(record.id)

      if (res.data && res.data.success) {
        ElMessage.success('预测记录及关联文件已成功删除')
        console.log('删除成功，服务器返回:', res.data)
        // 刷新历史记录列表
        fetchPredictionHistory()
      } else {
        ElMessage.error(res.data?.message || '删除预测记录失败')
        console.error('删除失败，服务器返回:', res.data)
      }
    } catch (e) {
      // 用户取消了删除操作
      if (e === 'cancel') {
        ElMessage.info('已取消删除操作')
      } else if (e !== 'close') {
        console.error('确认对话框错误:', e)
        ElMessage.error('操作过程中发生错误')
      }
    }
  } catch (error) {
    ElMessage.error(`删除预测记录失败: ${error.message}`)
    console.error('删除预测记录错误:', error)
  }
}

// 格式化数值显示
const formatNumber = (num) => {
  if (num === null || num === undefined) return 'N/A'
  // 如果数值很小，使用科学计数法
  if (Math.abs(num) < 0.001) {
    return num.toExponential(4)
  }
  // 否则保留4位小数
  return num.toFixed(4)
}

// 格式化日期时间
const formatDateTime = (dateTimeStr) => {
  if (!dateTimeStr) return 'N/A'
  return dateTimeStr
}

// 组件挂载时获取模型列表和历史记录
onMounted(() => {
  fetchAvailableModels()
  fetchPredictionHistory()
})
</script>

<template>
  <div class="tunnel-predict">
    <el-row :gutter="20">
      <!-- 左侧输入参数区域 -->
      <el-col :span="10">
        <el-card class="input-card">
          <template #header>
            <div class="card-header">
              <span>围岩参数输入</span>
            </div>
          </template>

          <el-form label-position="top">
            <!-- 模型选择 -->
            <el-form-item label="选择模型">
              <el-select v-model="modelData.modelPath" placeholder="请选择模型" style="width: 100%">
                <el-option v-for="model in modelData.availableModels" :key="model.path" :label="model.name"
                  :value="model.path" />
              </el-select>
              <div class="form-tip">从后端获取的可用模型列表</div>
            </el-form-item>

            <!-- 围岩参数输入 -->
            <el-form-item label="泊松比">
              <el-input-number v-model="inputParams.poissonRatio" :min="0.1" :max="0.5" :step="0.01" :precision="2"
                style="width: 100%" />
            </el-form-item>

            <el-form-item label="内摩擦角 (°)">
              <el-input-number v-model="inputParams.frictionAngle" :min="10" :max="50" :step="1" style="width: 100%" />
            </el-form-item>

            <el-form-item label="粘聚力 (MPa)">
              <el-input-number v-model="inputParams.cohesion" :min="0.1" :max="5" :step="0.1" :precision="2"
                style="width: 100%" />
            </el-form-item>

            <el-form-item label="剪胀角 (°)">
              <el-input-number v-model="inputParams.dilationAngle" :min="0" :max="20" :step="1" style="width: 100%" />
            </el-form-item>

            <el-form-item label="弹性模量 (MPa)">
              <el-input-number v-model="inputParams.elasticModulus" :min="0" :max="10000" :step="100"
                style="width: 100%" />
            </el-form-item>

            <!-- 预测按钮 -->
            <el-button type="primary" :loading="isLoading" @click="predict" style="width: 100%; margin-top: 20px;">
              开始预测
            </el-button>
          </el-form>
        </el-card>
      </el-col>

      <!-- 右侧预测结果区域 -->
      <el-col :span="14">
        <el-card class="result-card">
          <template #header>
            <div class="card-header">
              <span>位移预测结果</span>
              <div class="header-actions">
                <el-button type="primary" plain size="small" @click="openHistoryDialog">
                  <el-icon>
                    <Document />
                  </el-icon>
                  <span>历史记录</span>
                </el-button>
              </div>
            </div>
          </template>

          <div v-if="!showResult" class="no-result">
            <el-empty description="暂无预测结果" />
            <p class="no-result-tip">请在左侧输入围岩参数并点击"开始预测"按钮</p>
          </div>

          <div v-else class="result-content">
            <!-- 预测时间显示 -->
            <div class="prediction-time" v-if="predictionResult.predictionTime !== null">
              <el-icon>
                <Timer />
              </el-icon>
              <span>预测耗时: <strong>{{ predictionResult.predictionTime }}</strong> ms</span>
            </div>

            <!-- 位移预测结果 -->
            <el-row :gutter="20" class="result-section">
              <el-col :span="12">
                <div class="result-item">
                  <div class="result-label">拱顶下沉1 (mm):</div>
                  <div class="result-value">{{ formatNumber(predictionResult.crownSettlement1) }}</div>
                </div>
              </el-col>

              <el-col :span="12">
                <div class="result-item">
                  <div class="result-label">拱顶下沉2 (mm):</div>
                  <div class="result-value">{{ formatNumber(predictionResult.crownSettlement2) }}</div>
                </div>
              </el-col>

              <el-col :span="12">
                <div class="result-item">
                  <div class="result-label">周边收敛1 (mm):</div>
                  <div class="result-value">{{ formatNumber(predictionResult.convergence1) }}</div>
                </div>
              </el-col>

              <el-col :span="12">
                <div class="result-item">
                  <div class="result-label">周边收敛2 (mm):</div>
                  <div class="result-value">{{ formatNumber(predictionResult.convergence2) }}</div>
                </div>
              </el-col>

              <el-col :span="12">
                <div class="result-item">
                  <div class="result-label">拱脚下沉 (mm):</div>
                  <div class="result-value">{{ formatNumber(predictionResult.footSettlement) }}</div>
                </div>
              </el-col>
            </el-row>

            <!-- 模型说明 -->
            <div class="model-info">
              <el-alert type="info" :closable="false" show-icon>
                <template #title>
                  <span>模型信息</span>
                </template>
                <div class="model-info-content">
                  <p>当前使用模型: {{ modelData.modelPath }}</p>
                  <p>模型类型: 基于Transformer的时序预测模型</p>
                  <p>输入参数: 泊松比、内摩擦角、粘聚力、剪胀角、弹性模量</p>
                  <p>输出参数: 拱顶下沉1、拱顶下沉2、周边收敛1、周边收敛2、拱脚下沉</p>
                </div>
              </el-alert>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 历史预测记录对话框 -->
    <el-dialog v-model="historyDialogVisible" title="历史预测记录" width="80%" destroy-on-close>
      <div v-loading="loadingHistory">
        <div v-if="predictionHistory.length === 0" class="empty-history">
          <el-empty description="暂无历史预测记录" />
        </div>
        <el-table v-else :data="predictionHistory" style="width: 100%" max-height="500" border>
          <el-table-column prop="id" label="ID" width="60" />
          <el-table-column prop="timestamp" label="预测时间" width="160" />
          <el-table-column label="输入参数" width="280">
            <template #default="scope">
              <p><strong>泊松比:</strong> {{ scope.row.input_params.poisson_ratio }}</p>
              <p><strong>内摩擦角(°):</strong> {{ scope.row.input_params.friction_angle }}</p>
              <p><strong>粘聚力(MPa):</strong> {{ scope.row.input_params.cohesion }}</p>
              <p><strong>剪胀角(°):</strong> {{ scope.row.input_params.dilation_angle }}</p>
              <p><strong>弹性模量(MPa):</strong> {{ scope.row.input_params.elastic_modulus }}</p>
            </template>
          </el-table-column>
          <el-table-column label="预测结果">
            <template #default="scope">
              <p><strong>拱顶下沉1(mm):</strong> {{ formatNumber(scope.row.prediction_results.crown_settlement1) }}</p>
              <p><strong>拱顶下沉2(mm):</strong> {{ formatNumber(scope.row.prediction_results.crown_settlement2) }}</p>
              <p><strong>周边收敛1(mm):</strong> {{ formatNumber(scope.row.prediction_results.convergence1) }}</p>
              <p><strong>周边收敛2(mm):</strong> {{ formatNumber(scope.row.prediction_results.convergence2) }}</p>
              <p><strong>拱脚下沉(mm):</strong> {{ formatNumber(scope.row.prediction_results.foot_settlement) }}</p>
            </template>
          </el-table-column>
          <el-table-column prop="prediction_time_ms" label="预测耗时(ms)" width="120" />
          <el-table-column label="操作" width="160" fixed="right">
            <template #default="scope">
              <div class="action-buttons">
                <el-button type="primary" size="small" @click="viewHistoryDetail(scope.row)">
                  <el-icon>
                    <View />
                  </el-icon>
                  <span>查看</span>
                </el-button>
                <el-button type="danger" size="small" @click="deleteHistoryRecord(scope.row)">
                  <el-icon>
                    <Delete />
                  </el-icon>
                  <span>删除</span>
                </el-button>
              </div>
            </template>
          </el-table-column>
        </el-table>
      </div>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="historyDialogVisible = false">关闭</el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped lang="scss">
.tunnel-predict {
  padding: 20px;

  .card-header {
    font-size: 18px;
    font-weight: bold;
    display: flex;
    justify-content: space-between;
    align-items: center;

    .header-actions {
      display: flex;
      gap: 10px;
    }
  }

  .input-card,
  .result-card {
    height: 100%;
    min-height: 600px;
  }

  .form-tip {
    margin-top: 5px;
    font-size: 12px;
    color: #909399;
    font-style: italic;
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

  .result-content {
    padding: 10px;
  }

  .prediction-time {
    display: flex;
    align-items: center;
    padding: 10px;
    margin-bottom: 15px;
    background-color: #f0f9eb;
    border-radius: 4px;
    color: #67c23a;
    font-size: 14px;

    .el-icon {
      margin-right: 8px;
      font-size: 18px;
    }

    strong {
      font-weight: bold;
      margin: 0 3px;
    }
  }

  .section-title {
    font-size: 16px;
    font-weight: bold;
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 1px solid #ebeef5;
  }

  .result-section {
    margin-bottom: 30px;
    padding: 15px;
    background-color: #f8f9fa;
    border-radius: 8px;
    box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.05);
  }

  .result-item {
    display: flex;
    flex-direction: column;
    margin-bottom: 20px;

    .result-label {
      font-weight: bold;
      color: #606266;
      margin-bottom: 8px;
    }

    .result-value {
      font-size: 20px;
      font-weight: bold;
      color: #409EFF;
      text-align: center;
      padding: 10px;
      background-color: #f0f7ff;
      border-radius: 4px;
    }
  }

  .metrics-section {
    background-color: #f0f9eb;

    .metric-item {
      display: flex;
      flex-direction: column;
      margin-bottom: 15px;

      .metric-label {
        font-weight: bold;
        color: #606266;
        margin-bottom: 8px;
      }

      .metric-value {
        font-size: 16px;
        font-weight: bold;
        color: #67c23a;
        text-align: center;
        padding: 8px;
        background-color: #f0f9eb;
        border-radius: 4px;
        border: 1px solid #e1f3d8;
      }
    }
  }

  .model-info {
    margin-top: 20px;

    .model-info-content {
      padding: 10px 0;

      p {
        margin: 5px 0;
        line-height: 1.5;
      }
    }
  }

  .empty-history {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 300px;
  }

  .action-buttons {
    display: flex;
    gap: 8px;
    justify-content: center;
  }
}
</style>