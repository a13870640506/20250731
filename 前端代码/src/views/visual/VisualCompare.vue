<template>
  <div class="visual-compare">
    <el-card class="compare-card">
      <template #header>
        <div class="card-header">
          <span>结果对比</span>
        </div>
      </template>

      <div class="content-container">
        <!-- 左侧结果列表 -->
        <div class="result-list">
          <h3>预测结果列表</h3>
          <el-input v-model="searchQuery" placeholder="搜索结果..." prefix-icon="el-icon-search" clearable
            class="search-input" />

          <el-scrollbar height="500px">
            <div v-if="loading" class="loading-container">
              <el-skeleton :rows="5" animated />
            </div>
            <div v-else-if="filteredResults.length === 0" class="empty-container">
              <el-empty description="暂无预测结果" />
            </div>
            <div v-else class="result-items">
              <div v-for="result in filteredResults" :key="result.id" class="result-item"
                :class="{ 'active': selectedResult && selectedResult.id === result.id }" @click="selectResult(result)">
                <div class="result-item-header">
                  <span class="result-name">{{ result.model_name }}</span>
                  <el-tag size="small" type="info">{{ formatTimestamp(result.timestamp) }}</el-tag>
                </div>
                <div class="result-item-path">{{ result.path }}</div>
              </div>
            </div>
          </el-scrollbar>
        </div>

        <!-- 右侧结果详情 -->
        <div class="result-detail">
          <div v-if="!selectedResult" class="no-selection">
            <el-empty description="请从左侧选择预测结果" />
          </div>
          <div v-else-if="loadingDetail" class="loading-container">
            <el-skeleton :rows="10" animated />
          </div>
          <div v-else class="detail-content">
            <!-- 评估指标 -->
            <div class="metrics-summary">
              <el-row :gutter="30">
                <el-col :span="8">
                  <div class="metric-item">
                    <div class="metric-label">均方误差 (MSE):</div>
                    <div class="metric-value-inline">{{ resultDetail.metrics?.mse ?
                      formatScientificNumber(resultDetail.metrics.mse) : 'N/A' }}</div>
                  </div>
                </el-col>
                <el-col :span="8">
                  <div class="metric-item">
                    <div class="metric-label">均方根误差 (RMSE):</div>
                    <div class="metric-value-inline">{{ resultDetail.metrics?.rmse ?
                      formatScientificNumber(resultDetail.metrics.rmse) : 'N/A' }}</div>
                  </div>
                </el-col>
                <el-col :span="8">
                  <div class="metric-item">
                    <div class="metric-label">决定系数 (R2):</div>
                    <div class="metric-value-inline">{{ resultDetail.metrics?.r2 ? resultDetail.metrics.r2.toFixed(7) :
                      'N/A'
                      }}</div>
                  </div>
                </el-col>
              </el-row>
            </div>

            <!-- 数据表格 -->
            <div class="data-table">
              <h3>预测结果数据对比 <span class="total-count">(共 {{ resultDetail.total_rows || 0 }} 行)</span></h3>

              <el-table :data="resultDetail.results || []" stripe border style="width: 100%" height="550"
                :header-cell-style="{ background: '#f5f7fa', color: '#606266', fontWeight: 'bold' }">
                <el-table-column type="index" label="序号" width="80" align="center" />
                <el-table-column prop="predicted" label="预测值" align="center">
                  <template #default="scope">
                    <span>{{ formatNumber(scope.row.predicted) }}</span>
                  </template>
                </el-table-column>
                <el-table-column prop="actual" label="真实值" align="center">
                  <template #default="scope">
                    <span>{{ formatNumber(scope.row.actual) }}</span>
                  </template>
                </el-table-column>
                <el-table-column label="误差" align="center">
                  <template #default="scope">
                    <span :class="getErrorClass(scope.row.predicted, scope.row.actual)">
                      {{ formatError(scope.row.predicted, scope.row.actual) }}
                    </span>
                  </template>
                </el-table-column>
                <el-table-column label="相对误差 (%)" align="center">
                  <template #default="scope">
                    <span :class="getErrorClass(scope.row.predicted, scope.row.actual)">
                      {{ formatRelativeError(scope.row.predicted, scope.row.actual) }}
                    </span>
                  </template>
                </el-table-column>
              </el-table>

              <div class="table-footer">
                <p class="table-note">显示所有数据</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { getPredictionResultsService, getPredictionResultService } from '@/api/transformer'

// 响应式数据
const results = ref([])
const selectedResult = ref(null)
const resultDetail = ref({})
const loading = ref(false)
const loadingDetail = ref(false)
const searchQuery = ref('')

// 过滤结果列表
const filteredResults = computed(() => {
  if (!searchQuery.value) return results.value

  const query = searchQuery.value.toLowerCase()
  return results.value.filter(result =>
    result.name.toLowerCase().includes(query) ||
    result.model_name.toLowerCase().includes(query)
  )
})

// 格式化时间戳
const formatTimestamp = (timestamp) => {
  if (!timestamp) return ''

  // 格式: 20250720_231034 -> 2025-07-20 23:10:34
  const year = timestamp.substring(0, 4)
  const month = timestamp.substring(4, 6)
  const day = timestamp.substring(6, 8)
  const hour = timestamp.substring(9, 11)
  const minute = timestamp.substring(11, 13)
  const second = timestamp.substring(13, 15)

  return `${year}-${month}-${day} ${hour}:${minute}:${second}`
}

// 格式化科学计数
const formatScientificNumber = (num) => {
  if (num === null || num === undefined) return 'N/A'

  // 如果数值很小（小于0.0001），使用科学计数法
  if (Math.abs(num) < 0.0001) {
    return num.toExponential(7)
  }
  // 否则使用固定小数位
  return num.toFixed(7)
}

// 格式化数字
const formatNumber = (num) => {
  if (num === null || num === undefined) return 'N/A'
  return num.toFixed(7)
}

// 计算误差
const formatError = (predicted, actual) => {
  if (predicted === null || actual === null || predicted === undefined || actual === undefined) {
    return 'N/A'
  }

  const error = predicted - actual
  return error.toFixed(7)
}

// 计算相对误差
const formatRelativeError = (predicted, actual) => {
  if (predicted === null || actual === null || predicted === undefined || actual === undefined || actual === 0) {
    return 'N/A'
  }

  const relativeError = Math.abs((predicted - actual) / actual) * 100
  return relativeError.toFixed(2)
}

// 获取误差样式类
const getErrorClass = (predicted, actual) => {
  if (predicted === null || actual === null || predicted === undefined || actual === undefined) {
    return ''
  }

  const error = Math.abs(predicted - actual)
  const relativeError = actual !== 0 ? Math.abs(error / actual) : 0

  if (relativeError > 0.1) {
    return 'error-high'
  } else if (relativeError > 0.05) {
    return 'error-medium'
  } else {
    return 'error-low'
  }
}

// 选择结果
const selectResult = async (result) => {
  selectedResult.value = result
  await fetchResultDetail(result.id)
}

// 获取预测结果列表
const fetchResults = async () => {
  loading.value = true
  try {
    const res = await getPredictionResultsService()
    if (res.data && res.data.success) {
      results.value = res.data.data || []
    } else {
      ElMessage.error(res.data?.message || '获取预测结果列表失败')
    }
  } catch (error) {
    console.error('获取预测结果列表错误:', error)
    ElMessage.error(`获取预测结果列表失败: ${error.message}`)
  } finally {
    loading.value = false
  }
}

// 获取预测结果详情
const fetchResultDetail = async (resultId) => {
  loadingDetail.value = true
  resultDetail.value = {}

  try {
    const res = await getPredictionResultService(resultId)
    if (res.data && res.data.success) {
      resultDetail.value = res.data.data || {}
    } else {
      ElMessage.error(res.data?.message || '获取预测结果详情失败')
    }
  } catch (error) {
    console.error('获取预测结果详情错误:', error)
    ElMessage.error(`获取预测结果详情失败: ${error.message}`)
  } finally {
    loadingDetail.value = false
  }
}

// 组件挂载时获取数据
onMounted(() => {
  fetchResults()
})
</script>

<style scoped lang="scss">
.visual-compare {
  padding: 20px;

  .compare-card {
    min-height: 600px;

    .card-header {
      font-size: 18px;
      font-weight: bold;
    }

    .content-container {
      display: flex;
      gap: 20px;
      min-height: 550px;

      .result-list {
        flex: 0 0 300px;
        border-right: 1px solid #ebeef5;
        padding-right: 20px;

        h3 {
          margin-top: 0;
          margin-bottom: 15px;
          font-size: 16px;
        }

        .search-input {
          margin-bottom: 15px;
        }

        .result-items {
          .result-item {
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 10px;
            cursor: pointer;
            border: 1px solid #ebeef5;
            transition: all 0.3s;

            &:hover {
              background-color: #f5f7fa;
            }

            &.active {
              background-color: #ecf5ff;
              border-color: #409eff;
            }

            .result-item-header {
              display: flex;
              justify-content: space-between;
              align-items: center;
              margin-bottom: 5px;

              .result-name {
                font-weight: bold;
                font-size: 14px;
              }
            }

            .result-item-path {
              font-size: 12px;
              color: #909399;
              white-space: nowrap;
              overflow: hidden;
              text-overflow: ellipsis;
            }
          }
        }
      }

      .result-detail {
        flex: 1;

        .no-selection {
          height: 100%;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .detail-content {
          .metrics-summary {
            margin-bottom: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 8px;
            box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.05);

            .metric-item {
              display: flex;
              flex-direction: column;
              padding: 8px 0;
              max-width: 100%;

              .metric-label {
                font-weight: bold;
                color: #606266;
                margin-bottom: 5px;
                white-space: nowrap;
                width: 100%;
              }

              .metric-value-inline {
                font-weight: bold;
                color: #409EFF;
                text-align: center;
                padding: 5px;
                background-color: #f0f7ff;
                border-radius: 4px;
                overflow: hidden;
                text-overflow: ellipsis;
              }
            }
          }

          .data-table {
            h3 {
              margin-top: 0;
              margin-bottom: 15px;
              font-size: 16px;

              .total-count {
                font-size: 14px;
                color: #909399;
                font-weight: normal;
              }
            }

            .table-footer {
              margin-top: 10px;

              .table-note {
                font-size: 12px;
                color: #909399;
                font-style: italic;
              }
            }

            :deep(.error-high) {
              color: #f56c6c;
            }

            :deep(.error-medium) {
              color: #e6a23c;
            }

            :deep(.error-low) {
              color: #67c23a;
            }
          }
        }
      }
    }
  }

  .loading-container,
  .empty-container {
    padding: 20px;
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 200px;
  }
}
</style>
