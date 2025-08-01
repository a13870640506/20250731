<script setup>
import { ref } from 'vue'
import { computed } from 'vue'
import { onMounted } from 'vue'
import { baseURL } from '@/utils/request'

// 在页面实时更新时间
const currentTime = ref(new Date());
const formattedCurrentTime = computed(() => {
    // 自定义格式化函数
    const year = currentTime.value.getFullYear();
    const month = String(currentTime.value.getMonth() + 1).padStart(2, '0');
    const day = currentTime.value.getDate().toString().padStart(2, '0');
    const hours = currentTime.value.getHours().toString().padStart(2, '0');
    const minutes = currentTime.value.getMinutes().toString().padStart(2, '0');
    const seconds = currentTime.value.getSeconds().toString().padStart(2, '0');

    return `${year}年${month}月${day}日 ${hours}:${minutes}:${seconds}`;
});

// 在组件挂载时设置定时器
onMounted(() => {
    const timer = setInterval(() => {
        currentTime.value = new Date(); // 更新时间
    }, 1000);

    // 可以在这里返回一个清理函数，该函数将在组件卸载时执行
    return () => {
        clearInterval(timer);
    };
});

// 转换ip地址
const ipRegex = /(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})/;
const ipMatch = baseURL.match(ipRegex);
const currentIp = ipMatch ? ipMatch[0] : "未连接";

</script>

<template>
    <el-container class="layout-container">
        <el-aside width="200px">
            <div class="el-aside__logo">
                DT-Tunnel-TS
            </div>
            <el-menu active-text-color="#ffd04b" background-color="#232323" :default-active="$route.path"
                text-color="#fff" router>
                <el-menu-item index="/tunnel/model">
                    <el-icon>
                        <DataAnalysis />
                    </el-icon>
                    <span>模型训练</span>
                </el-menu-item>
                <el-menu-item index="/param/optimization">
                    <el-icon>
                        <Setting />
                    </el-icon>
                    <span>模型超参数优化</span>
                </el-menu-item>
                <el-menu-item index="/tunnel/predict">
                    <el-icon>
                        <Monitor />
                    </el-icon>
                    <span>隧道位移预测</span>
                </el-menu-item>
                <el-sub-menu index="/visual">
                    <template #title>
                        <el-icon>
                            <TrendCharts />
                        </el-icon>
                        <span>数据可视化</span>
                    </template>
                    <el-menu-item index="/visual/chart">
                        <span>训练结果详情</span>
                    </el-menu-item>
                    <el-menu-item index="/visual/compare">
                        <span>结果对比</span>
                    </el-menu-item>
                    <el-menu-item index="/visual/history">
                        <span>历史数据集</span>
                    </el-menu-item>
                </el-sub-menu>
                <el-menu-item index="/help/about">
                    <el-icon>
                        <InfoFilled />
                    </el-icon>
                    <span>帮助</span>
                </el-menu-item>
            </el-menu>
        </el-aside>
        <el-container>
            <el-header>
                <span></span>
                <div>
                    <el-text>
                        基于Transformer神经网络的隧道数字孪生系统-正演分析
                    </el-text>
                </div>
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <span class="time-spacing" style="margin-right: auto;">{{ formattedCurrentTime }}</span>
                    <div style="display: flex; align-items: center;">
                        <el-icon style="font-size: 28px; margin-right: 10px;margin-left: 10px;margin-top: 8px;">
                            <UserFilled />
                        </el-icon>
                        <el-icon style="font-size: 28px; margin-right: 10px;margin-top: 8px;">
                            <Bell />
                        </el-icon>
                    </div>
                </div>
            </el-header>

            <el-main>
                <router-view></router-view>
            </el-main>
            <el-footer>当前后端IP: {{ currentIp }} @2025 隧道数字孪生系统</el-footer>
        </el-container>
    </el-container>
</template>

<style lang="scss" scoped>
.layout-container {
    height: 100vh;

    .el-aside {
        background-color: #163D61;
        font-family: "Arial";

        &__logo {
            height: 68px;
            font-family: "Microsoft YaHei";
            line-height: 70px;
            text-align: center;
            font-size: 16px;
            color: #fff;
            font-weight: bold;
            border-bottom: 1.5px solid#545C64;
        }

        .el-menu {
            border-right: none;
            background-color: #163D61;
        }

        .el-menu-item {
            background-color: #163D61;
        }
    }

    .el-header {
        background-color: #61A2DE;
        display: flex;
        align-items: center;
        flex-direction: row;
        justify-content: space-between;
        height: 69px;

        .el-text {
            font-size: 28px;
            font-family: "Microsoft YaHei";
            color: #000;
            font-weight: bold;
        }

        .time-spacing {
            font-size: 18px;
            margin-top: 14px;
        }
    }

    .el-main {
        padding: 2px 20px;
    }

    .el-footer {
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        color: #666;
    }
}
</style>