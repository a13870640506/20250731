<template>
  <div class="visual-history">
    <el-card class="history-card">
      <template #header>
        <div class="card-header">
          <span>历史数据集</span>
        </div>
      </template>

      <div class="content-container">
        <!-- 左侧数据集列表 -->
        <div class="dataset-list">
          <h3>数据集列表</h3>
          <el-input v-model="searchQuery" placeholder="搜索数据集..." prefix-icon="el-icon-search" clearable
            class="search-input" />

          <div class="filter-options">
            <el-select v-model="categoryFilter" placeholder="类别" clearable class="filter-select">
              <el-option label="全部" value="" />
              <el-option label="训练集" value="训练集" />
              <el-option label="验证集" value="验证集" />
              <el-option label="测试集" value="测试集" />
            </el-select>

            <el-select v-model="typeFilter" placeholder="类型" clearable class="filter-select">
              <el-option label="全部" value="" />
              <el-option label="加速度" value="加速度" />
              <el-option label="位移" value="位移" />
            </el-select>
          </div>

          <el-scrollbar height="450px">
            <div v-if="loading" class="loading-container">
              <el-skeleton :rows="5" animated />
            </div>
            <div v-else-if="filteredDatasets.length === 0" class="empty-container">
              <el-empty description="暂无数据集" />
            </div>
            <div v-else class="dataset-items">
              <div v-for="dataset in filteredDatasets" :key="dataset.id" class="dataset-item"
                :class="{ 'active': selectedDataset && selectedDataset.id === dataset.id }"
                @click="selectDataset(dataset)">
                <div class="dataset-item-header">
                  <span class="dataset-name">{{ dataset.name }}</span>
                  <div class="dataset-tags">
                    <el-tag size="small" type="success">{{ dataset.category }}</el-tag>
                    <el-tag size="small" type="info">{{ dataset.type }}</el-tag>
                  </div>
                </div>
                <div class="dataset-item-info">
                  <span class="dataset-size">{{ formatSize(dataset.size) }}</span>
                  <span class="dataset-time">{{ dataset.modified_time }}</span>
                </div>
              </div>
            </div>
          </el-scrollbar>
        </div>

        <!-- 右侧数据集内容 -->
        <div class="dataset-detail">
          <div v-if="!selectedDataset" class="no-selection">
            <el-empty description="请从左侧选择数据集" />
          </div>
          <div v-else-if="loadingDetail" class="loading-container">
            <el-skeleton :rows="10" animated />
          </div>
          <div v-else class="detail-content">
            <div class="dataset-info">
              <h3>{{ selectedDataset.name }}</h3>
              <div class="dataset-meta">
                <div class="meta-item">
                  <span class="meta-label">类别:</span>
                  <el-tag size="small" type="success">{{ selectedDataset.category }}</el-tag>
                </div>
                <div class="meta-item">
                  <span class="meta-label">类型:</span>
                  <el-tag size="small" type="info">{{ selectedDataset.type }}</el-tag>
                </div>
                <div class="meta-item">
                  <span class="meta-label">大小:</span>
                  <span>{{ formatSize(selectedDataset.size) }}</span>
                </div>
                <div class="meta-item">
                  <span class="meta-label">修改时间:</span>
                  <span>{{ selectedDataset.modified_time }}</span>
                </div>
                <div class="meta-item">
                  <span class="meta-label">路径:</span>
                  <span class="dataset-path">{{ selectedDataset.path }}</span>
                </div>
              </div>
            </div>

            <!-- 数据表格 -->
            <div class="data-table">
              <h3>数据内容 <span class="total-count">(共 {{ datasetContent.total_rows || 0 }} 行)</span></h3>

              <el-table :data="datasetContent.rows || []" stripe border style="width: 100%" height="550"
                :header-cell-style="{ background: '#f5f7fa', color: '#606266', fontWeight: 'bold' }">
                <el-table-column type="index" label="序号" width="80" align="center" />
                <el-table-column v-for="header in datasetContent.headers || []" :key="header" :prop="header"
                  :label="header" align="center">
                  <template #default="scope">
                    <span>{{ formatTableValue(scope.row[header]) }}</span>
                  </template>
                </el-table-column>
              </el-table>

              <!-- 移除分页，显示所有数据 -->
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
import { getDatasetsService, getDatasetContentService } from '@/api/transformer'

// 响应式数据
const datasets = ref([])
const selectedDataset = ref(null)
const datasetContent = ref({})
const loading = ref(false)
const loadingDetail = ref(false)
const searchQuery = ref('')
const categoryFilter = ref('')
const typeFilter = ref('')
const currentPage = ref(1)
const pageSize = ref(10)

// 过滤数据集列表
const filteredDatasets = computed(() => {
  let filtered = datasets.value

  // 搜索过滤
  if (searchQuery.value) {
    const query = searchQuery.value.toLowerCase()
    filtered = filtered.filter(dataset =>
      dataset.name.toLowerCase().includes(query)
    )
  }

  // 类别过滤
  if (categoryFilter.value) {
    filtered = filtered.filter(dataset =>
      dataset.category === categoryFilter.value
    )
  }

  // 类型过滤
  if (typeFilter.value) {
    filtered = filtered.filter(dataset =>
      dataset.type === typeFilter.value
    )
  }

  return filtered
})

// 格式化文件大小
const formatSize = (bytes) => {
  if (bytes === 0) return '0 B'

  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))

  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

// 格式化表格值
const formatTableValue = (value) => {
  if (value === null || value === undefined) return 'N/A'

  if (typeof value === 'number') {
    // 如果数值很小（小于0.0001），使用科学计数法
    if (Math.abs(value) < 0.0001) {
      return value.toExponential(7)
    }
    // 否则使用固定小数位
    return value.toFixed(7)
  }

  return value
}

// 选择数据集
const selectDataset = async (dataset) => {
  selectedDataset.value = dataset
  currentPage.value = 1 // 重置页码
  await fetchDatasetContent(dataset.id)
}

// 处理页码变化
const handlePageChange = async (page) => {
  currentPage.value = page
  if (selectedDataset.value) {
    await fetchDatasetContent(selectedDataset.value.id, page)
  }
}

// 获取数据集列表
const fetchDatasets = async () => {
  loading.value = true
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
  } finally {
    loading.value = false
  }
}

// 获取数据集内容
const fetchDatasetContent = async (datasetId, page = 1) => {
  loadingDetail.value = true
  datasetContent.value = {}

  try {
    const res = await getDatasetContentService(datasetId, page, pageSize.value)
    if (res.data && res.data.success) {
      datasetContent.value = res.data.data || {}
    } else {
      ElMessage.error(res.data?.message || '获取数据集内容失败')
    }
  } catch (error) {
    console.error('获取数据集内容错误:', error)
    ElMessage.error(`获取数据集内容失败: ${error.message}`)
  } finally {
    loadingDetail.value = false
  }
}

// 组件挂载时获取数据
onMounted(() => {
  fetchDatasets()
})
</script>

<style scoped lang="scss">
.visual-history {
  padding: 20px;

  .history-card {
    min-height: 600px;

    .card-header {
      font-size: 18px;
      font-weight: bold;
    }

    .content-container {
      display: flex;
      gap: 20px;
      min-height: 550px;

      .dataset-list {
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

        .filter-options {
          display: flex;
          gap: 10px;
          margin-bottom: 15px;

          .filter-select {
            flex: 1;
          }
        }

        .dataset-items {
          .dataset-item {
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

            .dataset-item-header {
              display: flex;
              justify-content: space-between;
              align-items: center;
              margin-bottom: 5px;

              .dataset-name {
                font-weight: bold;
                font-size: 14px;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                max-width: 150px;
              }

              .dataset-tags {
                display: flex;
                gap: 5px;
              }
            }

            .dataset-item-info {
              display: flex;
              justify-content: space-between;
              font-size: 12px;
              color: #909399;

              .dataset-size {
                font-weight: bold;
              }
            }
          }
        }
      }

      .dataset-detail {
        flex: 1;

        .no-selection {
          height: 100%;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .detail-content {
          .dataset-info {
            margin-bottom: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 8px;
            box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.05);

            h3 {
              margin-top: 0;
              margin-bottom: 10px;
              font-size: 16px;
            }

            .dataset-meta {
              display: grid;
              grid-template-columns: repeat(3, 1fr);
              gap: 10px;

              .meta-item {
                display: flex;
                align-items: center;
                gap: 5px;

                .meta-label {
                  font-weight: bold;
                  color: #606266;
                }

                .dataset-path {
                  font-family: monospace;
                  color: #606266;
                  font-size: 12px;
                }
              }

              .meta-item:last-child {
                grid-column: span 3;
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

            .pagination-container {
              margin-top: 20px;
              display: flex;
              justify-content: center;
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
