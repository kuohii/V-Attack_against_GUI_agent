#!/bin/bash

# 激活 Conda 环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate Attack

# 获取当前时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 创建日志目录
mkdir -p logs

# 后台运行 ensemble 配置
echo "[$(date)] 启动 ensemble 攻击..." >> logs/attack_${TIMESTAMP}.log
nohup python V-Attack.py --config-name=ensemble > logs/ensemble_${TIMESTAMP}.log 2>&1 &
ENSEMBLE_PID=$!
echo "Ensemble PID: $ENSEMBLE_PID" >> logs/attack_${TIMESTAMP}.log

# 等待几秒避免资源冲突（可选）
sleep 2

# 后台运行 single 配置
echo "[$(date)] 启动 single 攻击..." >> logs/attack_${TIMESTAMP}.log
nohup python V-Attack.py --config-name=single > logs/single_${TIMESTAMP}.log 2>&1 &
SINGLE_PID=$!
echo "Single PID: $SINGLE_PID" >> logs/attack_${TIMESTAMP}.log

# 输出状态
echo "========================================"
echo "✅ 两个任务已在后台启动"
echo "📁 日志目录: logs/"
echo "🔹 Ensemble PID: $ENSEMBLE_PID"
echo "🔹 Single PID: $SINGLE_PID"
echo "========================================"
echo ""
echo "查看日志:"
echo "  tail -f logs/ensemble_${TIMESTAMP}.log"
echo "  tail -f logs/single_${TIMESTAMP}.log"
echo ""
echo "停止任务:"
echo "  kill $ENSEMBLE_PID $SINGLE_PID"