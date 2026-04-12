
export RUN_DIR="$SCRATCH_DISK/runs/swegym_pandas_qwen"

python analysis/test_eval_server.py \
    --predictions_path "$RUN_DIR/preds.json" \
    --server_url http://holy8a26112:8080 \
    --num_instances 20
