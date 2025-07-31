import request from '@/utils/request'

// 训练模型API
export const trainModelService = (data) => {
  return request({
    url: '/transformer/train',
    method: 'post',
    headers: {
      'Content-Type': 'multipart/form-data'
    },
    data
  })
}

// 预测API
export const predictService = (data) => {
  return request({
    url: '/transformer/predict',
    method: 'post',
    headers: {
      'Content-Type': 'multipart/form-data'
    },
    data
  })
}

// 上传数据集API
export const uploadDatasetService = (data) => {
  return request({
    url: '/transformer/upload',
    method: 'post',
    headers: {
      'Content-Type': 'multipart/form-data'
    },
    data
  })
}

// 超参数优化API
export const optimizeParamsService = (data) => {
  return request({
    url: '/transformer/optimize',
    method: 'post',
    headers: {
      'Content-Type': 'multipart/form-data'
    },
    data
  })
}

// 获取优化历史记录API
export const getOptimizationHistoryService = () => {
  return request({
    url: '/transformer/optimization_history',
    method: 'get'
  })
}

// 获取优化结果详情API
export const getOptimizationResultService = (optId) => {
  return request({
    url: `/transformer/optimization_result/${optId}`,
    method: 'get'
  })
}

// 获取数据可视化图表API
export const getChartDataService = (data) => {
  return request({
    url: '/transformer/visualize',
    method: 'post',
    headers: {
      'Content-Type': 'multipart/form-data'
    },
    data
  })
}

// 获取模型评估指标API
export const getModelMetricsService = (modelPath) => {
  return request({
    url: '/transformer/model_metrics',
    method: 'get',
    params: { model_path: modelPath }
  })
}

// 获取数据集列表API
export const getDatasetsService = () => {
  return request({
    url: '/transformer/datasets',
    method: 'get'
  })
} 