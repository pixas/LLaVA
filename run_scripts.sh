#!/bin/bash

# 检查是否提供了脚本路径参数
if [ "$#" -ne 1 ]; then
    echo "使用方法: $0 [要运行的脚本的路径]"
    exit 1
fi

# 要运行的脚本路径
script_path="$1"

# 检测条件
min_memory=32000  # 最小显存大小
min_cards=4       # 最少显卡数量

echo "开始监测显卡剩余显存..."

# 监测和执行的函数
monitor_and_execute() {
    while true; do
        echo "检查显卡状态..."
        # 获取每个GPU的剩余显存及其编号
        mapfile -t memory_info < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)

        # 筛选出满足条件的显卡编号和剩余显存
        declare -A valid_cards
        for info in "${memory_info[@]}"; do
            read -r index memory <<<"$info"
            if [ "$memory" -ge "$min_memory" ]; then
                valid_cards["$index"]=$memory
            fi
        done

        echo "找到 ${#valid_cards[@]} 张显卡的剩余显存大于等于 $min_memory MiB."

        # 检查是否有足够的显卡满足条件
        if [ "${#valid_cards[@]}" -ge "$min_cards" ]; then
            echo "满足条件，正在选择剩余显存最大的 $min_cards 张显卡..."

            # 获取剩余显存最大的4张卡的编号
            top_cards=($(for idx in "${!valid_cards[@]}"; do echo "$idx ${valid_cards[$idx]}"; done | sort -k2nr | awk '{print $1}' | head -n 4))
            for i in "${!top_cards[@]}"; do
                top_cards[$i]=${top_cards[$i]%,}
            done
            echo "选定的显卡编号为：${top_cards[*]}"

            # 将编号数组转换为逗号分隔的字符串
            cuda_devices=$(IFS=,; echo "${top_cards[*]}")
            echo "正在执行命令：CUDA_VISIBLE_DEVICES=$cuda_devices bash $script_path"

            # 执行命令
            CUDA_VISIBLE_DEVICES="$cuda_devices" bash "$script_path"

            echo "命令执行完毕，结束监测。"
            # 结束监测
            break
        else
            echo "未满足条件，等待1分钟后重新检查..."
        fi

        # 每1分钟检查一次
        sleep 60
    done
}

# 开始监测和执行过程
monitor_and_execute
