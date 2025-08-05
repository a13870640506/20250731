import axios from 'axios'


const baseURL = 'http://127.0.0.1:8000' // 后端服务器地址，可以根据Python Flask后端服务的实际地址进行修改


const instance = axios.create({
    baseURL,
    timeout: 30000  // 设置请求超时时间，当前的设置为30秒，可以根据需要调整
})


export default instance
export { baseURL }