<script setup>
import { ref, reactive, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { InfoFilled } from '@element-plus/icons-vue'
import { predictService } from '@/api/transformer'

// 创建响应式数据存储用户输入的内容
const formData = ref({
  model_path: "",
  input_dim: 1,
  output_dim: 1,
})

// 保存用户上传的文件
const selectedFiles = ref({
  input: null,
  output: null
});

const formFiles = ref({
  input: null,
  output: null
});

// 保存历史模型路径
const historyModelPaths = ref([]);

// 选中的历史模型路径
const selectedModelPath = ref(null);

// 当前激活的Tab
const activeTab = ref('tab1');

// 从本地存储加载历史模型路径
const loadHistoryModelPaths = () => {
  const savedPaths = localStorage.getItem('historyModelPaths');
  if (savedPaths) {
    try {
      historyModelPaths.value = JSON.parse(savedPaths);
    } catch (e) {
      console.error('解析历史模型路径失败:', e);
      historyModelPaths.value = [];
    }
  }
};

// 复制选中的历史模型路径
const copySelectedModelPath = () => {
  if (selectedModelPath.value) {
    navigator.clipboard.writeText(selectedModelPath.value).then(() => {
      ElMessage.success('历史模型路径已复制到剪贴板！');
    }).catch(err => {
      ElMessage.error(`复制失败: ${err}`);
    });
  } else {
    ElMessage.warning('未选择历史模型路径，无法复制。');
  }
};

// 处理文件选择变化的事件
const saveFile = (type, event) => {
  const file = event.target.files[0];
  if (file) {
    selectedFiles.value[type] = file;
    formFiles.value[type] = file;
    console.log(`已选择${type}文件:`, file.name);
  }
};

// 预测结果
const predictionResult = ref({
  actual: [],
  predicted: [],
  metrics: {
    mse: null,
    rmse: null,
    r2: null
  },
  timestamps: [],
  time_series_plot: null,
  scatter_plot: null,
  result_dir: null
});

// 是否显示结果
const showResult = ref(false);

// 是否有实际值可供比较
const hasActualValues = ref(false);

// 点击预测按钮
const isLoading = ref(false);
const predict = async () => {
  if (!formFiles.value.input) {
    ElMessage.warning('请上传输入文件');
    return;
  }

  if (!formData.value.model_path) {
    ElMessage.warning('请输入或选择模型路径');
    return;
  }

  try {
    isLoading.value = true;
    ElMessage.info('正在进行预测，请稍候...');

    // 模拟预测过程
    setTimeout(() => {
      // 模拟API响应
      const res = {
        data: {
          success: true,
          message: '预测成功',
          data: {
            predicted: Array.from({length: 100}, (_, i) => Math.sin(i/10) + Math.random()*0.2),
            actual: formFiles.value.output ? Array.from({length: 100}, (_, i) => Math.sin(i/10)) : [],
            metrics: {
              mse: 0.021,
              rmse: 0.145,
              r2: 0.932
            },
            timestamps: Array.from({length: 100}, (_, i) => i + 1),
            time_series_plot: 'base64_encoded_image_placeholder',
            scatter_plot: formFiles.value.output ? 'base64_encoded_image_placeholder' : null,
            result_dir: '/results/prediction_result_123'
          }
        }
      };

      if (res.data && res.data.success) {
        ElMessage.success('预测成功');
        
        predictionResult.value = res.data.data;
        hasActualValues.value = formFiles.value.output !== null;
        
        // 先设置显示结果标志
        showResult.value = true;
        
        // 保存模型路径到历史记录
        saveModelPathToHistory(formData.value.model_path);
      } else {
        ElMessage.error(res.data?.message || '预测失败');
      }
      
      isLoading.value = false;
    }, 2000);
    
  } catch (error) {
    ElMessage.error(`预测失败: ${error.message}`);
    console.error('预测错误:', error);
    isLoading.value = false;
  }
};

// 保存模型路径到历史记录
const saveModelPathToHistory = (path) => {
  if (!path) return;

  // 检查是否已存在
  const index = historyModelPaths.value.indexOf(path);
  if (index !== -1) {
    // 如果已存在，移到最前面
    historyModelPaths.value.splice(index, 1);
  }

  // 添加到历史记录开头
  historyModelPaths.value.unshift(path);

  // 限制历史记录数量
  if (historyModelPaths.value.length > 5) {
    historyModelPaths.value = historyModelPaths.value.slice(0, 5);
  }

  // 保存到本地存储
  localStorage.setItem('historyModelPaths', JSON.stringify(historyModelPaths.value));
};

// 选择历史模型路径
const selectModelPath = (path) => {
  formData.value.model_path = path;
};

// 复制结果路径
const copyResultPath = () => {
  if (predictionResult.value.result_dir) {
    navigator.clipboard.writeText(predictionResult.value.result_dir).then(() => {
      ElMessage.success('结果路径已复制到剪贴板！');
    }).catch(err => {
      ElMessage.error(`复制失败: ${err}`);
    });
  }
};

// 添加格式化科学计数的函数
const formatScientificNumber = (num) => {
  // 如果数值很小（小于0.0001），使用科学计数法
  if (Math.abs(num) < 0.0001) {
    return num.toExponential(7);
  }
  // 否则使用固定小数位
  return num.toFixed(7);
};

// 组件挂载时初始化
onMounted(() => {
  // 加载历史模型路径
  loadHistoryModelPaths();
});
</script>

<template>
  <div class="tunnel-predict">
    <el-row :gutter="20">
      <el-col :span="6">
        <el-card class="input-card">
          <template #header>
            <div class="card-header">
              <span>隧道位移预测</span>
            </div>
          </template>

          <el-form :model="formData" label-position="top">
            <el-form-item label="模型路径">
              <el-input v-model="formData.model_path" placeholder="请输入模型路径" />
            </el-form-item>

            <el-row :gutter="20" v-if="historyModelPaths.length > 0">
              <el-col :span="24">
                <el-form-item label="历史模型路径">
                  <el-select v-model="selectedModelPath" placeholder="选择历史模型路径" style="width: 100%">
                    <el-option v-for="(path, index) in historyModelPaths" :key="index" :label="path" :value="path" />
                  </el-select>
                  <div class="path-actions" v-if="selectedModelPath">
                    <el-button size="small" type="primary" @click="copySelectedModelPath" style="margin-top: 8px;">
                      复制路径
                    </el-button>
                    <el-button size="small" type="success" @click="selectModelPath(selectedModelPath)"
                      style="margin-top: 8px;">
                      使用此路径
                    </el-button>
                  </div>
                </el-form-item>
              </el-col>
            </el-row>

            <el-form-item label="输入文件 (时间序列数据)">
              <div class="file-upload">
                <input type="file" @change="(e) => saveFile('input', e)" accept=".csv" />
              </div>
            </el-form-item>

            <el-form-item label="输出文件 (可选，用于评估)">
              <div class="file-upload">
                <input type="file" @change="(e) => saveFile('output', e)" accept=".csv" />
              </div>
            </el-form-item>

            <el-form-item label="输入维度">
              <el-input v-model.number="formData.input_dim" type="number" placeholder="请输入维度" />
              <div class="form-tip">输入时间序列的维度，默认值为1</div>
            </el-form-item>

            <el-form-item label="输出维度">
              <el-input v-model.number="formData.output_dim" type="number" placeholder="请输入维度" />
              <div class="form-tip">输出时间序列的维度，默认值为1</div>
            </el-form-item>

            <el-button type="primary" :loading="isLoading" @click="predict" style="width: 100%;">开始预测</el-button>
          </el-form>
        </el-card>
      </el-col>

      <el-col :span="18">
        <el-card class="chart-card">
          <template #header>
            <div class="card-header">
              <div class="header-with-tabs">
                <span>预测结果</span>
                <!-- 将Tab标签移动到标题右侧 -->
                <el-tabs v-if="showResult" v-model="activeTab" type="card" class="result-tabs">
                  <el-tab-pane name="tab1" label="位移响应曲线"></el-tab-pane>
                  <el-tab-pane v-if="hasActualValues && predictionResult.scatter_plot" name="tab2"
                    label="散点对比图"></el-tab-pane>
                  <el-tab-pane name="tab3" label="结果信息"></el-tab-pane>
                </el-tabs>
              </div>
            </div>
          </template>

          <div v-if="!showResult" class="no-result">
            <el-empty description="暂无预测结果" />
            <p class="no-result-tip">请在左侧上传文件并点击"开始预测"按钮</p>
          </div>

          <template v-else>
            <!-- 评估指标放在图表之上 -->
            <div class="metrics-summary" v-if="hasActualValues">
              <el-row :gutter="30">
                <el-col :span="8">
                  <div class="metric-item">
                    <div class="metric-label">均方误差 (MSE):</div>
                    <div class="metric-value-inline">{{ predictionResult.metrics.mse ?
                      formatScientificNumber(predictionResult.metrics.mse) : 'N/A' }}</div>
                  </div>
                </el-col>
                <el-col :span="8">
                  <div class="metric-item">
                    <div class="metric-label">均方根误差 (RMSE):</div>
                    <div class="metric-value-inline">{{ predictionResult.metrics.rmse ?
                      formatScientificNumber(predictionResult.metrics.rmse) : 'N/A' }}</div>
                  </div>
                </el-col>
                <el-col :span="8">
                  <div class="metric-item">
                    <div class="metric-label">决定系数 (R2):</div>
                    <div class="metric-value-inline">{{ predictionResult.metrics.r2 ?
                      predictionResult.metrics.r2.toFixed(7) : 'N/A' }}</div>
                  </div>
                </el-col>
              </el-row>
            </div>

            <!-- 根据activeTab显示对应内容 -->
            <div v-if="activeTab === 'tab1'" class="plot-container">
              <div class="placeholder-image">
                <p>位移响应曲线图表将显示在这里</p>
              </div>
            </div>

            <div v-if="activeTab === 'tab2' && hasActualValues && predictionResult.scatter_plot"
              class="plot-container scatter-container">
              <div class="placeholder-image">
                <p>散点对比图将显示在这里</p>
              </div>
            </div>

            <div v-if="activeTab === 'tab3'" class="result-info-container">
              <h3>结果保存路径</h3>
              <el-input v-model="predictionResult.result_dir" readonly>
                <template #append>
                  <el-button @click="copyResultPath">复制</el-button>
                </template>
              </el-input>

              <div class="result-info-tips">
                <p>
                  <el-icon>
                    <InfoFilled />
                  </el-icon>
                  预测结果已保存到上述路径，包含CSV数据文件和图表图像。
                </p>
              </div>
            </div>
          </template>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped lang="scss">
.tunnel-predict {
  padding: 20px;
  position: relative;
  z-index: auto;

  .card-header {
    font-size: 18px;
    font-weight: bold;

    .header-with-tabs {
      display: flex;
      align-items: center;
      justify-content: space-between;

      span {
        flex-shrink: 0;
      }

      .result-tabs {
        margin-left: 20px;
        flex-grow: 1;

        :deep(.el-tabs__header) {
          margin: 0;
        }

        :deep(.el-tabs__nav-wrap) {
          padding-bottom: 0;
          display: flex;
          justify-content: flex-end;
        }

        :deep(.el-tabs__nav) {
          border: none;
        }

        :deep(.el-tabs__item) {
          height: 30px;
          line-height: 30px;
          font-size: 14px;
          border: 1px solid #dcdfe6;
          border-radius: 4px;
          margin-left: 5px;

          &.is-active {
            background-color: #409EFF;
            color: white;
            border-color: #409EFF;
          }
        }

        :deep(.el-tabs__content) {
          display: none;
        }
      }
    }
  }

  .input-card {
    height: 100%;
    overflow: visible;
  }

  .chart-card {
    height: 100%;
    overflow: visible;
  }

  .metrics-summary {
    margin-bottom: 1px;
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

  .plot-container {
    padding: 1px 10px;
    margin-top: 0;

    .placeholder-image {
      width: 100%;
      height: 400px;
      background-color: #f5f7fa;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #909399;
      font-style: italic;
    }
  }

  .scatter-container {
    display: flex;
    justify-content: center;

    .placeholder-image {
      max-width: 600px;
      width: 70%;
    }
  }

  .result-info-container {
    padding: 20px;

    h3 {
      margin-bottom: 15px;
      font-size: 16px;
      font-weight: bold;
    }

    .result-info-tips {
      margin-top: 20px;
      padding: 15px;
      background-color: #f8f9fa;
      border-radius: 8px;

      p {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #606266;

        .el-icon {
          color: #409EFF;
        }
      }
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

  .path-actions {
    display: flex;
    gap: 10px;
    justify-content: flex-start;
    flex-wrap: wrap;
  }
}
</style>