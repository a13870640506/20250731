import { createRouter, createWebHashHistory } from 'vue-router'

const router = createRouter({
  history: createWebHashHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      component: () => import('@/views/layout/layoutContainer.vue'),  // 侧边栏布局容器组件，包含导航菜单和主要内容区域
      redirect: '/tunnel/model',
      children: [ // 嵌套路由，用于在侧边栏布局容器中显示不同的页面内容
        {
          path: '/tunnel/model',
          component: () => import('@/views/tunnel/TunnelModel.vue')
        },
        {
          path: '/tunnel/predict',
          component: () => import('@/views/tunnel/TunnelPredict.vue')
        },
        {
          path: '/visual/chart',
          component: () => import('@/views/visual/VisualChart.vue')
        },
        {
          path: '/visual/compare',
          component: () => import('@/views/visual/VisualCompare.vue')
        },
        {
          path: '/visual/history',
          component: () => import('@/views/visual/VisualHistory.vue')
        },
        {
          path: '/param/optimization',
          component: () => import('@/views/param/ParamOptimization.vue')
        },
        {
          path: '/help/about',
          component: () => import('@/views/help/helpAbout.vue')
        }
      ]
    },
  ]
})

export default router
