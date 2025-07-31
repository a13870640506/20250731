<script setup>
import { ref, reactive, onMounted } from 'vue'
import { ElMessage } from 'element-plus'

// 创建响应式数据存储用户输入的内容
const formData = ref({
  data_file: null,
  chart_type: 'line',
  time_column: '',
  value_columns: []
})

// 文件上传状态
const isUploading = ref(false)

// 是否显示图表
const showChart = ref(false)

// 数据列选项
const columnOptions = ref([])

// 处理文件选择变化的事件
const handleFileChange = (event) => {
  const file = event.target.files[0]
  if (file) {
    formData.value.data_file = file
    console.log(`已选择文件:`, file.name)
    
    // 模拟读取CSV文件头获取列名
    setTimeout(() => {
      columnOptions.value = ['timestamp', 'value1', 'value2', 'value3', 'value4', 'value5']
      formData.value.time_column = 'timestamp'
      formData.value.value_columns = ['value1']
    }, 500)
  }
}

// 生成图表
const generateChart = () => {
  if (!formData.value.data_file) {
    ElMessage.warning('请上传数据文件')
    return
  }

  if (!formData.value.time_column) {
    ElMessage.warning('请选择时间列')
    return
  }

  if (formData.value.value_columns.length === 0) {
    ElMessage.warning('请至少选择一个数据列')
    return
  }

  isUploading.value = true
  
  // 模拟生成图表
  setTimeout(() => {
    showChart.value = true
    isUploading.value = false
    ElMessage.success('图表生成成功')
  }, 1000)
}
</script>

<template>
  <div class="visual-chart">
    <el-row :gutter="20">
      <el-col :span="6">
        <el-card class="input-card">
          <template #header>
            <div class="card-header">
              <span>数据可视化</span>
            </div>
          </template>

          <el-form :model="formData" label-position="top">
            <el-form-item label="数据文件">
              <div class="file-upload">
                <input type="file" @change="handleFileChange" accept=".csv" />
              </div>
            </el-form-item>

            <el-form-item label="图表类型">
              <el-select v-model="formData.chart_type" placeholder="选择图表类型" style="width: 100%">
                <el-option value="line" label="折线图" />
                <el-option value="bar" label="柱状图" />
                <el-option value="scatter" label="散点图" />
              </el-select>
            </el-form-item>

            <el-form-item label="时间列" v-if="columnOptions.length > 0">
              <el-select v-model="formData.time_column" placeholder="选择时间列" style="width: 100%">
                <el-option v-for="column in columnOptions" :key="column" :value="column" :label="column" />
              </el-select>
            </el-form-item>

            <el-form-item label="数据列" v-if="columnOptions.length > 0">
              <el-select v-model="formData.value_columns" multiple placeholder="选择数据列" style="width: 100%">
                <el-option v-for="column in columnOptions" :key="column" :value="column" :label="column" />
              </el-select>
            </el-form-item>

            <el-button type="primary" :loading="isUploading" @click="generateChart" style="width: 100%;">
              生成图表
            </el-button>
          </el-form>
        </el-card>
      </el-col>

      <el-col :span="18">
        <el-card class="chart-card">
          <template #header>
            <div class="card-header">
              <span>可视化结果</span>
            </div>
          </template>

          <div v-if="!showChart" class="no-chart">
            <el-empty description="暂无图表数据" />
            <p class="no-chart-tip">请在左侧上传数据文件并设置参数</p>
          </div>

          <div v-else class="chart-container">
            <div class="placeholder-chart">
              <p>图表将显示在这里</p>
              <p>选择的图表类型: {{ formData.chart_type }}</p>
              <p>时间列: {{ formData.time_column }}</p>
              <p>数据列: {{ formData.value_columns.join(', ') }}</p>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped lang="scss">
.visual-chart {
  padding: 20px;

  .card-header {
    font-size: 18px;
    font-weight: bold;
  }

  .input-card {
    height: 100%;
  }

  .chart-card {
    height: 100%;
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

  .no-chart {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 400px;

    .no-chart-tip {
      margin-top: 20px;
      color: #909399;
    }
  }

  .chart-container {
    height: 500px;
    
    .placeholder-chart {
      width: 100%;
      height: 100%;
      background-color: #f5f7fa;
      border-radius: 8px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      color: #909399;
      font-style: italic;
    }
  }
}
</style>