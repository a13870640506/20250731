<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import { ElMessage, ElLoading, ElMessageBox } from 'element-plus'
import { trainModelService, uploadDatasetService, getOptimizationHistoryService, getOptimizationResultService, getModelResultService, getRecentModelsService, getDatasetsService } from '@/api/transformer'
import { useRouter } from 'vue-router'
import { ArrowRight, View, CopyDocument, Download } from '@element-plus/icons-vue'
import { onBeforeMount } from 'vue'
import { format } from '@/utils/format'

// 当前激活的Tab
const activeTab = ref('dataPrep')
// 当前参数模式 - 贝叶斯优化或自定义
const paramMode = ref('bayesian')

// 创建router实例
const router = useRouter()

// 图表筛选相关
const selectedPlot = ref('all')

// 优化历史记录
const optimizationHistory = ref([])
// 选中的优化ID
const selectedOptimizationId = ref('')
// 优化结果详情
const optimizationResult = ref(null)
// 加载状态
const loadingOptHistory = ref(false)
const loadingOptResult = ref(false)

// 前往超参数优化页面
const goToOptimizationPage = () => {
  // 保存当前数据处理状态到localStorage
  if (dataProcessResult.value.data_path) {
    localStorage.setItem('tunnelModelDataPath', dataProcessResult.value.data_path)
    localStorage.setItem('tunnelModelDataResult', JSON.stringify(dataProcessResult.value))
    localStorage.setItem('tunnelModelShowDataResult', 'true')
  }

  router.push({
    path: '/param/optimization',
    query: {
      data_path: dataProcessResult.value.data_path
    }
  })
}

// 数据处理日志
const processLogs = ref([])

// 添加日志
const addLog = (message) => {
  const timestamp = new Date().toLocaleTimeString()
  processLogs.value.push(`[${timestamp}] ${message}`)
}

// 数据准备相关
const dataForm = ref({
  input_file: null,
  train_ratio: 0.7,
  val_ratio: 0.1,
  test_ratio: 0.2
})

// 更新测试集比例
const updateTestRatio = () => {
  dataForm.value.test_ratio = parseFloat((1 - dataForm.value.train_ratio - dataForm.value.val_ratio).toFixed(2))
}

// 数据上传状态
const isUploading = ref(false)

// 数据处理结果
const dataProcessResult = ref({
  train_size: 0,
  val_size: 0,
  test_size: 0,
  input_columns: [],
  output_columns: [],
  standardization_plots: [],
  data_path: ''
})

// 模型参数相关
const modelParams = ref({
  d_model: 256,
  nhead: 4,
  num_layers: 2,
  input_dim: 7,
  output_dim: 5,
  dropout: 0.05  // 添加dropout参数，默认值0.05
})

// 训练参数相关
const trainingParams = ref({
  epochs: 300,
  batch_size: 16,
  learning_rate: 8.564825241340346e-05,
  weight_decay: 2.3600012291720694e-07,
  model_save_path: ''
})


// 训练状态
const isTraining = ref(false)

// 训练结果
const trainingResult = ref({
  model_path: '',
  train_metrics: {
    loss: null,
    r2: null
  },
  val_metrics: {
    loss: null,
    r2: null
  },
  test_metrics: {
    loss: null,
    r2: null
  },
  loss_curve: null,
  model_params: null,
  training_params: null
})

// 最近训练的模型路径列表
const recentModelPaths = ref([])
const recentModelDates = ref([])

// 选中的模型路径
const selectedModelPath = ref('')
const selectedModelResult = ref(null)

// 数据集列表
const datasets = ref([])
const selectedDatasetPath = ref('')
const loadingDatasets = ref(false)

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
      return format(value)
  }
}

// 当前显示的结果（来自新训练或选择已有模型）
const currentResult = computed(() => {
  // 只有在明确加载了模型结果时才返回结果
  if (selectedModelResult.value) {
    return selectedModelResult.value
  } else if (showTrainingResult.value && trainingResult.value) {
    return trainingResult.value
  }
  return null
})

// 是否显示结果
const showDataResult = ref(false)
const showTrainingResult = ref(false)

// 加载优化历史
const loadOptimizationHistory = async () => {
  try {
    loadingOptHistory.value = true
    const res = await getOptimizationHistoryService()
    if (res.data && res.data.success) {
      optimizationHistory.value = res.data.data
    } else {
      ElMessage.warning('获取优化历史失败')
    }
  } catch (error) {
    ElMessage.error(`获取优化历史失败: ${error.message}`)
    console.error('获取优化历史错误:', error)
  } finally {
    loadingOptHistory.value = false
  }
}

// 加载优化结果详情
const loadOptimizationResult = async (optId) => {
  try {
    if (!optId) return

    loadingOptResult.value = true
    const res = await getOptimizationResultService(optId)

    if (res.data && res.data.success) {
      optimizationResult.value = res.data.data

      // 更新模型参数
      const bestParams = res.data.data.best_params
      modelParams.value.d_model = bestParams.d_model || 256
      modelParams.value.nhead = bestParams.nhead || 4
      modelParams.value.num_layers = bestParams.num_layers || 2

      // 更新训练参数
      trainingParams.value.learning_rate = bestParams.lr || 0.0001
      trainingParams.value.weight_decay = bestParams.weight_decay || 0.0001

      // 生成模型保存路径
      generateModelSavePath()

      ElMessage.success('已加载优化参数')
    } else {
      ElMessage.warning('获取优化结果详情失败')
    }
  } catch (error) {
    ElMessage.error(`获取优化结果详情失败: ${error.message}`)
    console.error('获取优化结果详情错误:', error)
  } finally {
    loadingOptResult.value = false
  }
}

// 查看优化结果
const viewOptimizationResult = (optId) => {
  router.push({
    path: '/param/optimization',
    query: { id: optId }
  })
}

// 生成模型保存路径
const generateModelSavePath = () => {
  // 根据超参数生成模型名称，与后端格式保持一致
  const d_model = modelParams.value.d_model
  // 确保学习率格式化为6位小数，与后端一致
  const lr = trainingParams.value.learning_rate.toFixed(6)
  const bs = trainingParams.value.batch_size

  trainingParams.value.model_save_path = `models/model_c${d_model}_lr${lr}_bs${bs}`
}

// 复制模型保存路径
const copyModelSavePath = () => {
  const path = trainingParams.value.model_save_path
  if (!path) {
    ElMessage.warning('请先生成模型保存路径')
    return
  }

  navigator.clipboard.writeText(path)
    .then(() => {
      ElMessage.success('已复制模型保存路径')
    })
    .catch(() => {
      ElMessage.error('复制失败，请手动复制')
    })
}

// 处理文件选择变化的事件
const handleFileChange = (type, event) => {
  const file = event.target.files[0]
  if (file) {
    dataForm.value[type] = file
    console.log(`已选择${type}:`, file.name)
  }
}

// 上传并处理数据集
const uploadDataset = async () => {
  if (!dataForm.value.input_file) {
    ElMessage.warning('请上传数据文件')
    return
  }

  try {
    isUploading.value = true
    processLogs.value = [] // 清空之前的日志
    addLog(`开始上传数据文件: ${dataForm.value.input_file.name}`)
    addLog(`数据集划分比例: 训练集=${dataForm.value.train_ratio}, 验证集=${dataForm.value.val_ratio}, 测试集=${dataForm.value.test_ratio}`)

    // 创建FormData对象用于上传文件
    const formDataToUpload = new FormData()
    formDataToUpload.append('input_file', dataForm.value.input_file)
    formDataToUpload.append('train_ratio', dataForm.value.train_ratio)
    formDataToUpload.append('val_ratio', dataForm.value.val_ratio)
    formDataToUpload.append('test_ratio', dataForm.value.test_ratio)

    const loading = ElLoading.service({
      lock: true,
      text: '正在处理数据...',
      background: 'rgba(0, 0, 0, 0.7)'
    })

    // 调用API上传数据
    try {
      addLog("正在处理数据，请稍候...")
      const res = await uploadDatasetService(formDataToUpload)

      if (res.data && res.data.success) {
        ElMessage.success('数据处理成功')
        dataProcessResult.value = res.data.data
        showDataResult.value = true

        // 更新模型参数中的输入输出维度
        modelParams.value.input_dim = res.data.data.input_dim || 7
        modelParams.value.output_dim = res.data.data.output_dim || 5

        // 生成模型保存路径
        generateModelSavePath()

        // 保存数据处理状态到localStorage
        if (dataProcessResult.value.data_path) {
          localStorage.setItem('tunnelModelDataPath', dataProcessResult.value.data_path)
          localStorage.setItem('tunnelModelDataResult', JSON.stringify(dataProcessResult.value))
          localStorage.setItem('tunnelModelShowDataResult', 'true')

          // 更新选中的数据集路径
          selectedDatasetPath.value = dataProcessResult.value.data_path
        }

        // 添加处理结果日志
        addLog("数据处理成功!")
        addLog(`训练集样本数: ${res.data.data.train_size}`)
        addLog(`验证集样本数: ${res.data.data.val_size}`)
        addLog(`测试集样本数: ${res.data.data.test_size}`)
        addLog(`输入维度: ${res.data.data.input_dim}`)
        addLog(`输出维度: ${res.data.data.output_dim}`)
        addLog("已生成标准化处理图表")
      } else {
        addLog(`错误: ${res.data?.message || '数据处理失败'}`)
        ElMessage.error(res.data?.message || '数据处理失败')
      }
    } catch (error) {
      addLog(`处理失败: ${error.message}`)
      ElMessage.error(`数据处理失败: ${error.message}`)
      console.error('数据处理错误:', error)
    } finally {
      loading.close()
      isUploading.value = false
    }
  } catch (error) {
    addLog(`上传失败: ${error.message}`)
    ElMessage.error(`上传失败: ${error.message}`)
    console.error('上传错误:', error)
    isUploading.value = false
  }
}

// 确认是否开始训练
const confirmTraining = (fromParamTab) => {
  if (fromParamTab) {
    // 从参数定义页面开始训练，检查是否选择了数据集
    if (!dataProcessResult.value.data_path) {
      ElMessage.warning('请先选择数据集')
      return
    }
  } else if (!showDataResult.value) {
    // 从其他页面开始训练，检查是否完成了数据准备
    ElMessageBox.confirm(
      '您还未完成数据准备，是否先前往数据准备标签页进行数据处理？',
      '提示',
      {
        confirmButtonText: '前往数据准备',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )
      .then(() => {
        // 切换到数据准备标签页
        activeTab.value = 'dataPrep'
      })
      .catch(() => {
        // 用户取消
      })
    return
  }

  // 确认开始训练
  ElMessageBox.confirm(
    `确定要使用当前参数开始训练模型吗？\n模型保存路径: ${trainingParams.value.model_save_path}`,
    '确认训练',
    {
      confirmButtonText: '确认',
      cancelButtonText: '取消',
      type: 'info'
    }
  )
    .then(() => {
      // 确认后开始训练
      startTraining(fromParamTab)
    })
    .catch(() => {
      // 用户取消
    })
}

// 提交表单开始训练
const startTraining = async (fromParamTab) => {
  try {
    isTraining.value = true
    // 重置选中的模型结果
    selectedModelPath.value = ''
    selectedModelResult.value = null

    // 如果是从参数定义标签页启动训练，则切换到训练结果标签页
    if (fromParamTab) {
      activeTab.value = 'modelTrain'
    }

    // 创建FormData对象用于训练参数
    const formDataToUpload = new FormData()
    formDataToUpload.append('data_path', dataProcessResult.value.data_path)
    formDataToUpload.append('epochs', trainingParams.value.epochs)
    formDataToUpload.append('batch_size', trainingParams.value.batch_size)
    formDataToUpload.append('learning_rate', trainingParams.value.learning_rate)
    formDataToUpload.append('weight_decay', trainingParams.value.weight_decay)
    formDataToUpload.append('d_model', modelParams.value.d_model)
    formDataToUpload.append('nhead', modelParams.value.nhead)
    formDataToUpload.append('num_layers', modelParams.value.num_layers)
    formDataToUpload.append('dropout', modelParams.value.dropout)
    formDataToUpload.append('mode', paramMode.value)
    formDataToUpload.append('input_dim', modelParams.value.input_dim)
    formDataToUpload.append('output_dim', modelParams.value.output_dim)
    if (trainingParams.value.model_save_path) {
      formDataToUpload.append('model_save_path', trainingParams.value.model_save_path)
    }

    const loading = ElLoading.service({
      lock: true,
      text: '模型训练中，请稍候...',
      background: 'rgba(0, 0, 0, 0.7)'
    })

    // 调用API训练模型
    try {
      const res = await trainModelService(formDataToUpload)

      if (res.data && res.data.success) {
        ElMessage.success('模型训练成功')
        trainingResult.value = res.data.data
        showTrainingResult.value = true


        // 如果后端返回模型目录，自动选中该路径
        if (trainingResult.value.model_dir) {
          selectedModelPath.value = trainingResult.value.model_dir
        }


        // 训练成功后，刷新最近模型列表
        loadRecentModels()
      } else {
        ElMessage.error(res.data?.message || '模型训练失败')
      }
    } catch (error) {
      ElMessage.error(`训练失败: ${error.message}`)
      console.error('训练错误:', error)
    } finally {
      loading.close()
      isTraining.value = false
    }
  } catch (error) {
    ElMessage.error(`训练失败: ${error.message}`)
    console.error('训练错误:', error)
    isTraining.value = false
  }
}

// 加载最近训练的模型列表
const loadRecentModels = async () => {
  try {
    console.log("开始加载最近训练记录...")
    const res = await getRecentModelsService()

    if (res.data && res.data.success) {
      recentModelPaths.value = res.data.data.paths || []
      recentModelDates.value = res.data.data.dates || []
      console.log("成功加载最近训练记录:", recentModelPaths.value)

      if (recentModelPaths.value.length === 0) {
        console.warn("没有找到训练记录，请检查模型目录是否存在训练结果")
      }
    } else {
      console.error('获取最近模型列表失败:', res.data?.message)
      ElMessage.warning('获取最近训练记录失败')
    }
  } catch (error) {
    console.error('获取最近模型列表错误:', error)
    ElMessage.error('获取最近训练记录出错')
  }
}

// 加载数据集列表
const loadDatasets = async () => {
  try {
    loadingDatasets.value = true
    console.log("开始加载数据集列表...")
    const res = await getDatasetsService()

    if (res.data && res.data.success) {
      datasets.value = res.data.data || []
      console.log("成功加载数据集列表:", datasets.value)

      // 如果当前已有数据集路径，则保持选中
      if (dataProcessResult.value.data_path) {
        selectedDatasetPath.value = dataProcessResult.value.data_path
      } else if (datasets.value.length > 0) {
        // 否则默认选择第一个数据集
        selectedDatasetPath.value = datasets.value[0].path
      }
    } else {
      console.error('获取数据集列表失败:', res.data?.message)
      ElMessage.warning('获取数据集列表失败')
    }
  } catch (error) {
    console.error('获取数据集列表错误:', error)
    ElMessage.error('获取数据集列表失败')
  } finally {
    loadingDatasets.value = false
  }
}

// 加载指定模型路径的结果
const loadModelResult = async (modelPath) => {
  if (!modelPath) {
    selectedModelResult.value = null
    return
  }

  try {
    const loading = ElLoading.service({
      lock: true,
      text: '加载模型结果...',
      background: 'rgba(0, 0, 0, 0.7)'
    })

    console.log("正在加载模型结果:", modelPath)
    const res = await getModelResultService(modelPath)

    if (res.data && res.data.success) {
      console.log("模型结果加载成功:", res.data.data)
      selectedModelResult.value = res.data.data

      // 确保模型路径正确显示
      if (!selectedModelResult.value.model_path) {
        selectedModelResult.value.model_path = modelPath
      }

      // 确保始终显示结果
      showTrainingResult.value = true

      ElMessage.success('模型结果加载成功')
    } else {
      ElMessage.warning('获取模型结果失败: ' + (res.data?.message || '未知错误'))
      selectedModelResult.value = null
    }

    loading.close()
  } catch (error) {
    ElMessage.error(`获取模型结果失败: ${error.message}`)
    console.error('获取模型结果错误:', error)
    selectedModelResult.value = null
    loading.close()
  }
}

// 复制选中的模型路径
const copySelectedModelPath = () => {
  if (!selectedModelPath.value) {
    ElMessage.warning('请先选择模型路径')
    return
  }

  navigator.clipboard.writeText(selectedModelPath.value)
    .then(() => {
      ElMessage.success('已复制模型路径')
    })
    .catch(() => {
      ElMessage.error('复制失败，请手动复制')
    })
}

// 选择数据集后更新数据处理结果
const handleDatasetSelect = async (datasetPath) => {
  if (!datasetPath) return

  // 查找选中的数据集信息
  const selectedDataset = datasets.value.find(dataset => dataset.path === datasetPath)
  if (!selectedDataset) return

  // 更新数据处理结果
  dataProcessResult.value = {
    data_path: selectedDataset.path,
    train_size: selectedDataset.info['训练集样本数'] || 0,
    val_size: selectedDataset.info['验证集样本数'] || 0,
    test_size: selectedDataset.info['测试集样本数'] || 0,
    input_dim: selectedDataset.info['输入维度'] || 7,
    output_dim: selectedDataset.info['输出维度'] || 5,
    standardization_plots: selectedDataset.plots || []
  }

  // 更新模型参数中的输入输出维度
  modelParams.value.input_dim = dataProcessResult.value.input_dim
  modelParams.value.output_dim = dataProcessResult.value.output_dim

  // 显示数据结果
  showDataResult.value = true

  // 生成模型保存路径
  generateModelSavePath()

  ElMessage.success(`已选择数据集: ${selectedDataset.date}`)
}

// 从localStorage恢复数据处理状态
const restoreDataProcessState = () => {
  try {
    const savedDataPath = localStorage.getItem('tunnelModelDataPath')
    const savedDataResult = localStorage.getItem('tunnelModelDataResult')
    const savedShowDataResult = localStorage.getItem('tunnelModelShowDataResult')

    if (savedDataPath && savedDataResult && savedShowDataResult === 'true') {
      console.log("从localStorage恢复数据处理状态")
      dataProcessResult.value = JSON.parse(savedDataResult)
      selectedDatasetPath.value = savedDataPath
      showDataResult.value = true

      // 更新模型参数中的输入输出维度
      if (dataProcessResult.value.input_dim) {
        modelParams.value.input_dim = dataProcessResult.value.input_dim
      }
      if (dataProcessResult.value.output_dim) {
        modelParams.value.output_dim = dataProcessResult.value.output_dim
      }
    }
  } catch (error) {
    console.error("恢复数据处理状态失败:", error)
  }
}

// 组件挂载前加载优化历史和最近模型
onBeforeMount(async () => {
  // 并行加载优化历史、最近模型列表和数据集列表
  await Promise.all([
    loadOptimizationHistory(),
    loadRecentModels(),
    loadDatasets()
  ])

  console.log("已加载最近模型列表:", recentModelPaths.value)

  // 从localStorage恢复数据处理状态
  restoreDataProcessState()

  // 如果有模型记录，自动选择第一个并加载
  if (recentModelPaths.value && recentModelPaths.value.length > 0) {
    selectedModelPath.value = recentModelPaths.value[0]
    if (activeTab.value === 'modelTrain') {
      // 自动加载第一个模型的结果
      await loadModelResult(selectedModelPath.value)
    }
  }

  // 监听参数变化，自动更新模型保存路径
  generateModelSavePath()
})
</script>

<template>
  <div class="tunnel-model">
    <el-card>
      <el-tabs v-model="activeTab">
        <!-- 数据准备标签页 -->
        <el-tab-pane name="dataPrep" label="数据准备">
          <el-row :gutter="20">
            <el-col :span="10">
              <el-form :model="dataForm" label-position="top">
                <el-form-item label="数据文件 (CSV格式)">
                  <div class="file-upload">
                    <input type="file" @change="(e) => handleFileChange('input_file', e)" accept=".csv" />
                    <div class="form-tip">请上传数据文件，包含输入特征和输出标签</div>
                  </div>
                </el-form-item>

                <el-divider>数据集划分比例</el-divider>

                <el-row :gutter="10">
                  <el-col :span="8">
                    <el-form-item label="训练集">
                      <el-input-number v-model="dataForm.train_ratio" :min="0.5" :max="0.9" :step="0.05" :precision="2"
                        @change="updateTestRatio" />
                    </el-form-item>
                  </el-col>
                  <el-col :span="8">
                    <el-form-item label="验证集">
                      <el-input-number v-model="dataForm.val_ratio" :min="0.05" :max="0.2" :step="0.05" :precision="2"
                        @change="updateTestRatio" />
                    </el-form-item>
                  </el-col>
                  <el-col :span="8">
                    <el-form-item label="测试集">
                      <el-input-number v-model="dataForm.test_ratio" :min="0.1" :max="0.3" :step="0.05" :precision="2"
                        disabled />
                      <div class="form-tip">自动计算: {{ (1 - dataForm.train_ratio - dataForm.val_ratio).toFixed(2) }}</div>
                    </el-form-item>
                  </el-col>
                </el-row>

                <el-button type="primary" :loading="isUploading" @click="uploadDataset" style="width: 100%;">
                  上传并处理数据
                </el-button>
              </el-form>

              <el-divider>数据处理日志</el-divider>
              <div class="log-container">
                <div v-if="processLogs.length === 0" class="empty-log">
                  <span>暂无日志信息</span>
                </div>
                <div v-else class="log-content">
                  <div v-for="(log, index) in processLogs" :key="index" class="log-item">
                    {{ log }}
                  </div>
                </div>
              </div>
            </el-col>

            <el-col :span="14">
              <div v-if="!showDataResult" class="no-result">
                <el-empty description="暂无数据处理结果" />
                <p class="no-result-tip">请上传数据文件并点击"上传并处理数据"按钮</p>
              </div>

              <div v-else class="data-result">
                <div class="data-result-header">
                  <div class="data-info-container">
                    <el-descriptions title="数据集信息" :column="3" border>
                      <el-descriptions-item label="训练集样本数">{{ dataProcessResult.train_size }}</el-descriptions-item>
                      <el-descriptions-item label="验证集样本数">{{ dataProcessResult.val_size }}</el-descriptions-item>
                      <el-descriptions-item label="测试集样本数">{{ dataProcessResult.test_size }}</el-descriptions-item>
                    </el-descriptions>
                  </div>
                </div>

                <!-- 移动到右上角位置 -->
                <div class="action-button-container">
                  <el-button type="primary" @click="goToOptimizationPage" class="optimization-button">
                    前往超参数优化页面
                    <el-icon class="el-icon--right">
                      <ArrowRight />
                    </el-icon>
                  </el-button>
                </div>



                <el-divider>标准化处理前后对比</el-divider>

                <div class="plot-filter-container">
                  <div class="plot-filter">
                    <span class="filter-label">筛选图表：</span>
                    <el-select v-model="selectedPlot" placeholder="选择图表" style="width: 250px">
                      <el-option label="全部图表" value="all" />
                      <el-option v-for="(plot, index) in dataProcessResult.standardization_plots" :key="index"
                        :label="plot.name" :value="plot.name" />
                    </el-select>
                  </div>
                </div>

                <div class="standardization-plots">
                  <div v-for="(plot, index) in dataProcessResult.standardization_plots" :key="index" class="plot-item"
                    v-show="selectedPlot === 'all' || selectedPlot === plot.name">
                    <div class="plot-image-container">
                      <img :src="`data:image/png;base64,${plot.image}`" :alt="plot.title" class="plot-image" />
                    </div>
                    <div class="plot-footer">
                      <div></div> <!-- 空div占位，保持flex布局 -->
                      <a v-if="plot.image_path"
                        :href="`http://127.0.0.1:8000/download?path=${encodeURIComponent(plot.image_path)}`"
                        target="_blank" class="download-link">
                        下载高清图
                      </a>
                    </div>
                  </div>
                </div>
              </div>
            </el-col>
          </el-row>
        </el-tab-pane>

        <!-- 参数定义标签页 -->
        <el-tab-pane name="paramDef" label="参数定义">
          <el-tabs v-model="paramMode" type="card" class="param-mode-tabs">
            <!-- 贝叶斯优化参数标签页 -->
            <el-tab-pane name="bayesian" label="使用贝叶斯优化参数">
              <el-row :gutter="20">
                <el-col :span="24">
                  <div class="opt-select-container">
                    <!-- 选择数据集卡片 -->
                    <el-card shadow="hover" class="opt-select-card compact-card">
                      <div class="inline-form-layout">
                        <div class="inline-form-label">选择数据集:</div>
                        <div class="inline-form-content">
                          <el-select v-model="selectedDatasetPath" placeholder="选择数据集" filterable clearable
                            :loading="loadingDatasets" @change="handleDatasetSelect">
                            <el-option v-for="dataset in datasets" :key="dataset.path"
                              :label="`${dataset.date} (训练集: ${dataset.info['训练集样本数'] || '?'}, 验证集: ${dataset.info['验证集样本数'] || '?'}, 测试集: ${dataset.info['测试集样本数'] || '?'})`"
                              :value="dataset.path" />
                          </el-select>
                          <span class="inline-form-tip">选择已处理的数据集</span>
                        </div>
                      </div>
                    </el-card>

                    <!-- 选择优化结果卡片 -->
                    <el-card shadow="hover" class="opt-select-card compact-card" style="margin-top: 10px;">
                      <div class="inline-form-layout">
                        <div class="inline-form-label">选择优化结果:</div>
                        <div class="inline-form-content">
                          <el-select v-model="selectedOptimizationId" placeholder="选择优化结果" filterable clearable
                            :loading="loadingOptHistory" @change="loadOptimizationResult">
                            <el-option v-for="item in optimizationHistory" :key="item.id"
                              :label="`${item.date} (最佳参数: d_model=${item.best_params.d_model}, nhead=${item.best_params.nhead}, num_layers=${item.best_params.num_layers}, lr=${item.best_params.lr}, weight_decay=${item.best_params.weight_decay})`"
                              :value="item.id" />
                          </el-select>
                          <span class="inline-form-tip">选择已完成的优化结果</span>

                          <div class="inline-form-buttons">
                            <el-button type="primary" size="small" :icon="View" :disabled="!selectedOptimizationId"
                              @click="viewOptimizationResult(selectedOptimizationId)">
                              查看详情
                            </el-button>
                            <el-button type="success" size="small" :icon="ArrowRight" @click="goToOptimizationPage">
                              新优化
                            </el-button>
                          </div>
                        </div>
                      </div>
                    </el-card>
                  </div>
                </el-col>
              </el-row>

              <el-divider>当前优化参数</el-divider>

              <el-row :gutter="20" v-loading="loadingOptResult">
                <el-col :span="12">
                  <el-card class="param-card" shadow="hover">
                    <template #header>
                      <div class="param-header">
                        <span>模型参数</span>
                      </div>
                    </template>

                    <el-form :model="modelParams" label-position="top">
                      <el-form-item>
                        <template #label>
                          <div class="form-label-with-tip">
                            <span>模型维度 (d_model)</span>
                            <span class="inline-tip">Transformer模型的嵌入维度</span>
                          </div>
                        </template>
                        <el-input-number v-model="modelParams.d_model" :min="64" :max="512" :step="64"
                          style="width: 100%" disabled />
                      </el-form-item>

                      <el-form-item>
                        <template #label>
                          <div class="form-label-with-tip">
                            <span>注意力头数 (nhead)</span>
                            <span class="inline-tip">多头自注意力机制中的头数</span>
                          </div>
                        </template>
                        <el-input-number v-model="modelParams.nhead" :min="1" :max="8" style="width: 100%" disabled />
                      </el-form-item>

                      <el-form-item>
                        <template #label>
                          <div class="form-label-with-tip">
                            <span>Transformer层数</span>
                            <span class="inline-tip">Transformer编码器层数</span>
                          </div>
                        </template>
                        <el-input-number v-model="modelParams.num_layers" :min="1" :max="6" style="width: 100%"
                          disabled />
                      </el-form-item>



                      <el-form-item>
                        <template #label>
                          <div class="form-label-with-tip">
                            <span>输入维度</span>
                            <span class="inline-tip">由数据准备阶段自动设置</span>
                          </div>
                        </template>
                        <el-input-number v-model="modelParams.input_dim" :min="1" :max="20" style="width: 100%"
                          disabled />
                      </el-form-item>

                      <el-form-item>
                        <template #label>
                          <div class="form-label-with-tip">
                            <span>输出维度</span>
                            <span class="inline-tip">由数据准备阶段自动设置</span>
                          </div>
                        </template>
                        <el-input-number v-model="modelParams.output_dim" :min="1" :max="10" style="width: 100%"
                          disabled />
                      </el-form-item>
                    </el-form>
                  </el-card>
                </el-col>

                <el-col :span="12">
                  <el-card class="param-card" shadow="hover">
                    <template #header>
                      <div class="param-header">
                        <span>训练参数</span>
                      </div>
                    </template>

                    <el-form :model="trainingParams" label-position="top">
                      <el-form-item>
                        <template #label>
                          <div class="form-label-with-tip">
                            <span>最大训练轮次 (Epochs)</span>
                            <span class="inline-tip">训练过程中数据集的最大迭代次数</span>
                          </div>
                        </template>
                        <el-input-number v-model="trainingParams.epochs" :min="10" :max="1000" style="width: 100%" />
                      </el-form-item>

                      <el-form-item>
                        <template #label>
                          <div class="form-label-with-tip">
                            <span>批次大小 (Batch Size)</span>
                            <span class="inline-tip">每次参数更新使用的样本数</span>
                          </div>
                        </template>
                        <el-input-number v-model="trainingParams.batch_size" :min="1" :max="128" style="width: 100%" />
                      </el-form-item>

                      <el-form-item>
                        <template #label>
                          <div class="form-label-with-tip">
                            <span>学习率 (Learning Rate)</span>
                            <span class="inline-tip">参数更新步长，使用优化结果的最佳学习率</span>
                          </div>
                        </template>
                        <el-input-number v-model="trainingParams.learning_rate" :min="1e-7" :max="1e-2" :step="1e-5"
                          :precision="7" :controls="false" style="width: 100%" disabled />
                      </el-form-item>

                      <el-form-item>
                        <template #label>
                          <div class="form-label-with-tip">
                            <span>权重衰减 (Weight Decay)</span>
                            <span class="inline-tip">L2正则化系数，使用优化结果的最佳权重衰减</span>
                          </div>
                        </template>
                        <el-input-number v-model="trainingParams.weight_decay" :min="1e-9" :max="1e-3" :step="1e-7"
                          :precision="9" :controls="false" style="width: 100%" disabled />
                      </el-form-item>

                      <el-form-item>
                        <template #label>
                          <div class="form-label-with-tip">
                            <span>模型保存路径</span>
                            <span class="inline-tip">根据参数自动生成的模型保存路径</span>
                          </div>
                        </template>
                        <div class="path-input-group">
                          <el-input v-model="trainingParams.model_save_path" disabled />
                          <el-button type="primary" :icon="CopyDocument" @click="copyModelSavePath"></el-button>
                        </div>
                      </el-form-item>

                      <el-button type="primary" size="large" style="width: 100%; margin-top: 20px;"
                        @click="confirmTraining(true)">
                        开始训练
                      </el-button>
                    </el-form>
                  </el-card>
                </el-col>
              </el-row>
            </el-tab-pane>

            <!-- 自定义参数标签页 -->
            <el-tab-pane name="custom" label="自定义参数">
              <!-- 选择数据集卡片 -->
              <el-row :gutter="20">
                <el-col :span="24">
                  <el-card shadow="hover" class="opt-select-card compact-card">
                    <div class="inline-form-layout">
                      <div class="inline-form-label">选择数据集:</div>
                      <div class="inline-form-content">
                        <el-select v-model="selectedDatasetPath" placeholder="选择数据集" filterable clearable
                          :loading="loadingDatasets" @change="handleDatasetSelect">
                          <el-option v-for="dataset in datasets" :key="dataset.path"
                            :label="`${dataset.date} (训练集: ${dataset.info['训练集样本数'] || '?'}, 验证集: ${dataset.info['验证集样本数'] || '?'}, 测试集: ${dataset.info['测试集样本数'] || '?'})`"
                            :value="dataset.path" />
                        </el-select>
                        <span class="inline-form-tip">选择已处理的数据集</span>
                      </div>
                    </div>
                  </el-card>
                </el-col>
              </el-row>

              <el-divider>模型参数设置</el-divider>

              <el-row :gutter="20">
                <el-col :span="12">
                  <el-card class="param-card" shadow="hover">
                    <template #header>
                      <div class="param-header">
                        <span>模型参数</span>
                      </div>
                    </template>

                    <el-form :model="modelParams" label-position="top">
                      <el-form-item>
                        <template #label>
                          <div class="form-label-with-tip">
                            <span>模型维度 (d_model)</span>
                            <span class="inline-tip">Transformer模型的嵌入维度</span>
                          </div>
                        </template>
                        <el-select v-model="modelParams.d_model" style="width: 100%" @change="generateModelSavePath">
                          <el-option :value="64" label="64" />
                          <el-option :value="128" label="128" />
                          <el-option :value="256" label="256" />
                          <el-option :value="512" label="512" />
                        </el-select>
                      </el-form-item>

                      <el-form-item>
                        <template #label>
                          <div class="form-label-with-tip">
                            <span>注意力头数 (nhead)</span>
                            <span class="inline-tip">多头自注意力机制中的头数</span>
                          </div>
                        </template>
                        <el-select v-model="modelParams.nhead" style="width: 100%" @change="generateModelSavePath">
                          <el-option :value="1" label="1" />
                          <el-option :value="2" label="2" />
                          <el-option :value="4" label="4" />
                          <el-option :value="8" label="8" />
                        </el-select>
                      </el-form-item>

                      <el-form-item>
                        <template #label>
                          <div class="form-label-with-tip">
                            <span>Transformer层数</span>
                            <span class="inline-tip">Transformer编码器层数</span>
                          </div>
                        </template>
                        <el-input-number v-model="modelParams.num_layers" :min="1" :max="6" style="width: 100%"
                          @change="generateModelSavePath" />
                      </el-form-item>



                      <el-form-item>
                        <template #label>
                          <div class="form-label-with-tip">
                            <span>输入维度</span>
                            <span class="inline-tip">由数据准备阶段自动设置</span>
                          </div>
                        </template>
                        <el-input-number v-model="modelParams.input_dim" :min="1" :max="20" style="width: 100%"
                          disabled />
                      </el-form-item>

                      <el-form-item>
                        <template #label>
                          <div class="form-label-with-tip">
                            <span>输出维度</span>
                            <span class="inline-tip">由数据准备阶段自动设置</span>
                          </div>
                        </template>
                        <el-input-number v-model="modelParams.output_dim" :min="1" :max="10" style="width: 100%"
                          disabled />
                      </el-form-item>
                    </el-form>
                  </el-card>
                </el-col>

                <el-col :span="12">
                  <el-card class="param-card" shadow="hover">
                    <template #header>
                      <div class="param-header">
                        <span>训练参数</span>
                      </div>
                    </template>

                    <el-form :model="trainingParams" label-position="top">
                      <el-form-item>
                        <template #label>
                          <div class="form-label-with-tip">
                            <span>最大训练轮次 (Epochs)</span>
                            <span class="inline-tip">训练过程中数据集的最大迭代次数</span>
                          </div>
                        </template>
                        <el-input-number v-model="trainingParams.epochs" :min="10" :max="1000" style="width: 100%" />
                      </el-form-item>

                      <el-form-item>
                        <template #label>
                          <div class="form-label-with-tip">
                            <span>批次大小 (Batch Size)</span>
                            <span class="inline-tip">每次参数更新使用的样本数</span>
                          </div>
                        </template>
                        <el-input-number v-model="trainingParams.batch_size" :min="1" :max="128" style="width: 100%"
                          @change="generateModelSavePath" />
                      </el-form-item>

                      <el-form-item>
                        <template #label>
                          <div class="form-label-with-tip">
                            <span>学习率 (Learning Rate)</span>
                            <span class="inline-tip">参数更新步长，建议范围：1e-5 ~ 1e-3</span>
                          </div>
                        </template>
                        <el-input-number v-model="trainingParams.learning_rate" :min="1e-7" :max="1e-2" :step="1e-5"
                          :precision="7" :controls="false" style="width: 100%" @change="generateModelSavePath" />
                      </el-form-item>

                      <el-form-item>
                        <template #label>
                          <div class="form-label-with-tip">
                            <span>权重衰减 (Weight Decay)</span>
                            <span class="inline-tip">L2正则化系数，防止过拟合</span>
                          </div>
                        </template>
                        <el-input-number v-model="trainingParams.weight_decay" :min="1e-9" :max="1e-3" :step="1e-7"
                          :precision="9" :controls="false" style="width: 100%" @change="generateModelSavePath" />
                      </el-form-item>

                      <el-form-item>
                        <template #label>
                          <div class="form-label-with-tip">
                            <span>模型保存路径</span>
                            <span class="inline-tip">根据参数自动生成的模型保存路径</span>
                          </div>
                        </template>
                        <div class="path-input-group">
                          <el-input v-model="trainingParams.model_save_path" />
                          <el-button type="primary" :icon="CopyDocument" @click="copyModelSavePath"></el-button>
                        </div>
                      </el-form-item>

                      <el-button type="primary" size="large" style="width: 100%; margin-top: 20px;"
                        @click="confirmTraining(true)">
                        开始训练
                      </el-button>
                    </el-form>
                  </el-card>
                </el-col>
              </el-row>
            </el-tab-pane>
          </el-tabs>
        </el-tab-pane>

        <!-- 模型训练标签页 -->
        <el-tab-pane name="modelTrain" label="训练结果">
          <el-row :gutter="20">
            <el-col :span="6">
              <el-card class="train-card" shadow="hover">
                <template #header>
                  <div class="train-header">
                    <span>模型选择</span>
                  </div>
                </template>

                <el-divider>最近训练记录</el-divider>
                <div class="recent-models">
                  <el-select v-model="selectedModelPath" placeholder="选择模型路径" style="width: 100%; margin-bottom: 10px;"
                    @change="loadModelResult" filterable>
                    <el-option v-for="(path, index) in recentModelPaths" :key="index" :label="path" :value="path">
                      <span class="model-path-option">{{ path }}</span>
                      <span class="model-path-date text-muted">{{ recentModelDates[index] }}</span>
                    </el-option>
                  </el-select>
                </div>

                <el-divider>模型路径</el-divider>
                <div class="path-input-group">
                  <el-input v-model="selectedModelPath" placeholder="输入或粘贴模型路径" clearable />
                  <el-button type="primary" :icon="CopyDocument" @click="copySelectedModelPath"></el-button>
                </div>

                <el-button type="primary" style="width: 100%; margin-top: 15px;"
                  @click="loadModelResult(selectedModelPath)">
                  加载模型结果
                </el-button>
              </el-card>
            </el-col>

            <el-col :span="18">
              <el-row :gutter="20">
                <el-col :span="12">
                  <el-card class="loss-curve-card">
                    <template #header>
                      <div class="result-header">
                        <span>训练损失曲线</span>
                      </div>
                    </template>

                    <div v-if="!currentResult" class="no-result">
                      <el-empty description="暂无训练结果" />
                      <p class="no-result-tip">请选择模型路径并点击"加载模型结果"</p>
                    </div>

                    <template v-else>
                      <div class="loss-curve">
                        <img v-if="currentResult.loss_curve" :src="`data:image/png;base64,${currentResult.loss_curve}`"
                          alt="训练损失曲线" class="curve-image" />
                        <div v-else class="placeholder-charts">
                          <p class="chart-placeholder">该模型没有可用的损失曲线图</p>
                        </div>
                      </div>
                    </template>
                  </el-card>
                </el-col>

                <el-col :span="12">
                  <el-card class="metrics-card">
                    <template #header>
                      <div class="result-header">
                        <span>评估指标与训练参数</span>
                      </div>
                    </template>

                    <div v-if="!currentResult" class="no-result">
                      <el-empty description="暂无训练结果" />
                      <p class="no-result-tip">请选择模型路径并点击"加载模型结果"</p>
                    </div>

                    <template v-else>
                      <el-descriptions title="模型信息" :column="1" border size="small">
                        <el-descriptions-item label="模型路径">{{ currentResult.model_path }}</el-descriptions-item>
                      </el-descriptions>

                      <el-divider>评估指标</el-divider>
                      <el-tabs type="card">
                        <el-tab-pane label="训练集">
                          <div class="metric-item">
                            <span class="metric-label">损失 (MSE):</span>
                            <span class="metric-value">{{ currentResult.train_metrics?.loss ?
                              formatNumber(currentResult.train_metrics.loss, 'mse') : 'N/A' }}</span>
                          </div>
                          <div class="metric-item">
                            <span class="metric-label">R² 系数:</span>
                            <span class="metric-value">{{ currentResult.train_metrics?.r2 ?
                              formatNumber(currentResult.train_metrics.r2, 'r2') : 'N/A' }}</span>
                          </div>
                          <div class="metric-item">
                            <span class="metric-label">MAPE:</span>
                            <span class="metric-value">{{ currentResult.train_metrics?.mape ?
                              formatNumber(currentResult.train_metrics.mape, 'mape') : 'N/A' }}</span>
                          </div>
                        </el-tab-pane>
                        <el-tab-pane label="验证集">
                          <div class="metric-item">
                            <span class="metric-label">损失 (MSE):</span>
                            <span class="metric-value">{{ currentResult.val_metrics?.loss ?
                              formatNumber(currentResult.val_metrics.loss, 'mse') : 'N/A' }}</span>
                          </div>
                          <div class="metric-item">
                            <span class="metric-label">R² 系数:</span>
                            <span class="metric-value">{{ currentResult.val_metrics?.r2 ?
                              formatNumber(currentResult.val_metrics.r2, 'r2') : 'N/A' }}</span>
                          </div>
                          <div class="metric-item">
                            <span class="metric-label">MAPE:</span>
                            <span class="metric-value">{{ currentResult.val_metrics?.mape ?
                              formatNumber(currentResult.val_metrics.mape, 'mape') : 'N/A' }}</span>
                          </div>
                        </el-tab-pane>
                        <el-tab-pane label="测试集">
                          <div class="metric-item">
                            <span class="metric-label">损失 (MSE):</span>
                            <span class="metric-value">{{ currentResult.test_metrics?.loss ?
                              formatNumber(currentResult.test_metrics.loss, 'mse') : 'N/A' }}</span>
                          </div>
                          <div class="metric-item">
                            <span class="metric-label">R² 系数:</span>
                            <span class="metric-value">{{ currentResult.test_metrics?.r2 ?
                              formatNumber(currentResult.test_metrics.r2, 'r2') : 'N/A' }}</span>
                          </div>
                          <div class="metric-item">
                            <span class="metric-label">MAPE:</span>
                            <span class="metric-value">{{ currentResult.test_metrics?.mape ?
                              formatNumber(currentResult.test_metrics.mape, 'mape') : 'N/A' }}</span>
                          </div>
                        </el-tab-pane>
                      </el-tabs>

                      <el-divider>训练参数</el-divider>
                      <div class="training-params">
                        <div class="param-item">
                          <span class="param-label">模型维度 (d_model):</span>
                          <span class="param-value">{{ currentResult.model_params?.d_model || 'N/A' }}</span>
                        </div>
                        <div class="param-item">
                          <span class="param-label">注意力头数 (nhead):</span>
                          <span class="param-value">{{ currentResult.model_params?.nhead || 'N/A' }}</span>
                        </div>
                        <div class="param-item">
                          <span class="param-label">Transformer层数:</span>
                          <span class="param-value">{{ currentResult.model_params?.num_layers || 'N/A' }}</span>
                        </div>
                        <div class="param-item">
                          <span class="param-label">学习率:</span>
                          <span class="param-value">{{ currentResult.training_params?.learning_rate ?
                            formatNumber(currentResult.training_params.learning_rate) : 'N/A' }}</span>
                        </div>
                        <div class="param-item">
                          <span class="param-label">权重衰减:</span>
                          <span class="param-value">{{ currentResult.training_params?.weight_decay ?
                            formatNumber(currentResult.training_params.weight_decay) : 'N/A' }}</span>
                        </div>
                        <div class="param-item">
                          <span class="param-label">批次大小:</span>
                          <span class="param-value">{{ currentResult.training_params?.batch_size || 'N/A' }}</span>
                        </div>
                        <div class="param-item">
                          <span class="param-label">训练轮次:</span>
                          <span class="param-value">{{ currentResult.training_params?.epochs || 'N/A' }}</span>
                        </div>
                      </div>
                    </template>
                  </el-card>
                </el-col>
              </el-row>
            </el-col>
          </el-row>
        </el-tab-pane>
      </el-tabs>
    </el-card>
  </div>
</template>

<style scoped lang="scss">
.tunnel-model {
  padding: 15px;

  /* 适当减小Tab标签页上部的页边距 */
  :deep(.el-tabs__content) {
    padding-top: 5px;
  }

  /* 调整Tab标签头部的边距 */
  :deep(.el-tabs__header) {
    margin-bottom: 15px;
  }

  /* 内部Tab标签样式调整 */
  .param-mode-tabs {
    :deep(.el-tabs__header) {
      margin-bottom: 20px;
    }
  }

  .card-header {
    font-size: 18px;
    font-weight: bold;
  }

  .log-container {
    height: 200px;
    border: 1px solid #dcdfe6;
    border-radius: 4px;
    background-color: #f5f7fa;
    overflow-y: auto;
    padding: 10px;
    margin-top: 10px;
    font-family: monospace;

    .empty-log {
      display: flex;
      justify-content: center;
      align-items: center;
      height: 100%;
      color: #909399;
      font-style: italic;
    }

    .log-content {
      .log-item {
        padding: 3px 0;
        border-bottom: 1px dashed #ebeef5;
        font-size: 13px;

        &:last-child {
          border-bottom: none;
        }
      }
    }
  }

  /* 优化选择相关样式 */
  .opt-select-container {
    margin-bottom: 20px;
  }

  .opt-select-card {
    background-color: #f8fafc;

    &.compact-card {
      padding: 12px;
    }
  }

  .opt-header {
    font-weight: bold;
    font-size: 16px;
  }

  .opt-content {
    display: flex;
    flex-direction: column;
    gap: 15px;

    .opt-select {
      flex: 1;
    }

    .opt-buttons {
      display: flex;
      justify-content: flex-end;
      gap: 10px;
    }
  }

  /* 行内表单布局 */
  .inline-form-layout {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
  }

  .inline-form-label {
    font-weight: 600;
    color: #303133;
    white-space: nowrap;
    min-width: 100px;
  }

  .inline-form-content {
    flex: 1;
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;

    .el-select {
      min-width: 400px;
      width: 100%;
    }
  }

  /* 确保下拉选项能够显示完整内容 */
  :deep(.el-select-dropdown__item) {
    white-space: normal;
    height: auto;
    padding: 8px 12px;
    line-height: 1.5;
  }

  .inline-form-tip {
    color: #909399;
    font-size: 12px;
    font-style: italic;
    white-space: nowrap;
  }

  .inline-form-buttons {
    display: flex;
    gap: 8px;
    margin-left: auto;
  }



  /* 模型保存路径输入框组样式 */
  .path-input-group {
    display: flex;
    align-items: center;
    gap: 10px;

    .el-input {
      flex: 1;
    }
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
    }
  }

  .file-upload {
    display: flex;
    flex-direction: column;
    gap: 5px;

    input[type="file"] {
      padding: 8px;
      border: 1px solid #dcdfe6;
      border-radius: 4px;
      background-color: #f5f7fa;

      &:hover {
        border-color: #c0c4cc;
      }
    }
  }

  .form-tip {
    margin-top: 5px;
    font-size: 12px;
    color: #909399;
    font-style: italic;
  }

  .form-label-with-tip {
    display: flex;
    justify-content: space-between;
    align-items: center;
    width: 100%;

    .inline-tip {
      font-size: 12px;
      color: #909399;
      font-style: italic;
      margin-left: 10px;
      text-align: right;
    }
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

  .data-result {
    padding: 10px;
    position: relative;

    .data-result-header {
      display: flex;
      justify-content: space-between;
      margin-bottom: 20px;

      .data-info-container {
        width: 100%;
      }
    }

    .action-button-container {
      position: absolute;
      top: 20px;
      right: 20px;
      z-index: 10;

      .optimization-button {
        font-size: 14px;
        padding: 8px 16px;
        border-radius: 20px;
        box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
      }
    }

    .feature-list {
      margin-bottom: 15px;

      h4 {
        margin-bottom: 10px;
        color: #303133;
      }

      .feature-tag {
        margin-right: 8px;
        margin-bottom: 8px;
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
          justify-content: space-between;
          align-items: center;

          .plot-title {
            font-size: 14px;
            color: #606266;
            margin: 0;
          }

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
  }

  .param-card,
  .train-card,
  .result-card {
    height: 100%;
  }

  .param-header,
  .train-header,
  .result-header,
  .metric-header {
    font-weight: bold;
    font-size: 16px;
  }

  .train-info {
    margin-bottom: 20px;
    padding: 15px;
    background-color: #f8f9fa;
    border-radius: 4px;

    p {
      margin: 8px 0;
      color: #606266;
    }
  }

  .train-tip {
    margin-top: 15px;
  }

  .metric-item {
    display: flex;
    justify-content: space-between;
    margin-bottom: 10px;

    .metric-label {
      color: #606266;
    }

    .metric-value {
      font-weight: bold;
      color: #409EFF;
    }
  }

  .loss-curve {
    margin-top: 20px;
    text-align: center;

    .curve-image {
      max-width: 100%;
      border-radius: 4px;
      box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
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

  .recent-models {
    display: flex;
    flex-direction: column;
    gap: 10px;
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

  .training-params {
    margin-top: 10px;
  }

  .param-item {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
  }

  .param-label {
    color: #606266;
  }

  .param-value {
    font-weight: bold;
    color: #303133;
  }

  .loss-curve-card,
  .metrics-card {
    height: 100%;
    min-height: 450px;
  }

  .text-muted {
    color: #909399;
  }
}
</style>