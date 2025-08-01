import { createRouter, createWebHashHistory } from 'vue-router'

const router = createRouter({
  history: createWebHashHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      component: () => import('@/views/layout/layoutContainer.vue'),
      redirect: '/tunnel/model',
      children: [
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
