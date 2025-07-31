import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import '@/assets/main.scss'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'
import 'element-plus/dist/index.css'

// import ElementPlus from 'element-plus'
// import 'element-plus/dist/index.css'
import * as echarts from 'echarts'



const app = createApp(App)

app.provide('echarts', echarts)
app.use(router)
// app.use(ElementPlus)
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
    app.component(key, component)
}
app.mount('#app')
