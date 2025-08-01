import request from '@/utils/request'

// 数据集上传和处理
export function uploadDatasetService(data) {
  return request({
    url: '/transformer/upload',
    method: 'post',
    data,
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  })
}

// 获取最近训练的模型列表
export function getRecentModelsService() {
  return request({
    url: '/transformer/recent_models',
    method: 'get'
  })
}

// 获取最新训练的模型
export function getLatestModelService() {
  return request({
    url: '/transformer/latest_model',
    method: 'get'
  })
}

// 获取模型结果
export function getModelResultService(modelPath) {
  return request({
    url: '/transformer/model_result',
    method: 'get',
    params: { model_path: modelPath }
  })
}

// 获取数据集列表
export function getDatasetsService() {
  return request({
    url: '/transformer/datasets',
    method: 'get'
  })
}

// 获取历史数据集列表
export function getDatasetHistoryService() {
  return request({
    url: '/transformer/dataset_history',
    method: 'get'
  })
}

// 获取数据集详情
export function getDatasetDetailService(datasetId) {
  return request({
    url: '/transformer/dataset_detail',
    method: 'get',
    params: { id: datasetId }
  })
}

// 训练模型
export function trainModelService(data) {
  return request({
    url: '/transformer/train',
    method: 'post',
    data
  })
}

// 模型预测
export function predictService(data) {
  return request({
    url: '/transformer/predict',
    method: 'post',
    data
  })
}

// 超参数优化
export function optimizeParamsService(data) {
  return request({
    url: '/transformer/optimize',
    method: 'post',
    data
  })
}

// 获取优化历史记录
export function getOptimizationHistoryService() {
  return request({
    url: '/transformer/optimization_history',
    method: 'get'
  })
}

// 获取优化结果详情
export function getOptimizationResultService(optId) {
  return request({
    url: `/transformer/optimization_result/${optId}`,
    method: 'get'
  })
}