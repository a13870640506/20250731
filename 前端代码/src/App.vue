<script setup>
import zh from 'element-plus/es/locale/lang/zh-cn.mjs'
</script>

<template>

  <el-config-provider :locale="zh">
    <router-view /> <!-- 使用 router-view 渲染路由组件 -->
  </el-config-provider>

</template>


<style scoped></style>
