#!/bin/bash
# Параллельный запуск 4 задач на одной GPU.
# Использование:
#   bash scripts/launch_4tasks.sh
#
# Перед запуском должен быть готов кэш YOLO bbox (data/skater_bboxes.json).
# Если нет — запустить любую одну задачу для построения кэша:
#   python scripts/train_single_task.py --task fall

set -e

cd "$(dirname "$0")/.."

if [ ! -f "data/skater_bboxes.json" ]; then
    echo "ERROR: data/skater_bboxes.json не найден."
    echo "Запусти сначала любую одну задачу для построения YOLO кэша:"
    echo "  python scripts/train_single_task.py --task fall"
    exit 1
fi

mkdir -p logs

# expandable_segments снижает фрагментацию памяти при параллельных аллокациях
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Launching 4 tasks in parallel on GPU $VIDEOMAE_CUDA_DEVICE (default: 1)..."

python scripts/train_single_task.py --task jump  > logs/jump.log  2>&1 &
PID_JUMP=$!
python scripts/train_single_task.py --task rot   > logs/rot.log   2>&1 &
PID_ROT=$!
python scripts/train_single_task.py --task under > logs/under.log 2>&1 &
PID_UNDER=$!
python scripts/train_single_task.py --task fall  > logs/fall.log  2>&1 &
PID_FALL=$!

echo "PIDs: jump=$PID_JUMP rot=$PID_ROT under=$PID_UNDER fall=$PID_FALL"
echo ""
echo "Мониторинг:"
echo "  tail -f logs/jump.log"
echo "  watch -n 5 nvidia-smi"
echo ""
echo "Остановка всех:"
echo "  kill $PID_JUMP $PID_ROT $PID_UNDER $PID_FALL"
echo ""

wait
echo "All 4 tasks complete."
