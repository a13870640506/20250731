import { dayjs } from 'element-plus'

export const formatTime = (time) => dayjs(time).format('YYYY年MM月DD日 HH:mm:ss')

/**
 * 格式化数字，保留适当的小数位数
 * @param {number} value - 要格式化的数字
 * @param {number} [minDigits=6] - 最小小数位数
 * @param {number} [maxDigits=10] - 最大小数位数
 * @returns {string} 格式化后的数字字符串
 */
export const format = (value, minDigits = 6, maxDigits = 10) => {
  if (value === undefined || value === null) return 'N/A'
  
  // 如果是字符串，尝试转换为数字
  if (typeof value === 'string') {
    value = parseFloat(value)
  }
  
  // 如果不是数字，返回原值
  if (isNaN(value)) return value
  
  // 如果是整数或大于1的数字，直接返回
  if (Number.isInteger(value) || Math.abs(value) >= 1) {
    return value.toString()
  }
  
  // 对于小数，根据值的大小动态调整小数位数
  const absValue = Math.abs(value)
  let digits = minDigits
  
  // 对于非常小的数字，增加小数位数
  if (absValue < 0.0000001) {
    digits = maxDigits
  } else if (absValue < 0.000001) {
    digits = 9
  } else if (absValue < 0.00001) {
    digits = 8
  } else if (absValue < 0.0001) {
    digits = 7
  }
  
  return value.toFixed(digits)
}