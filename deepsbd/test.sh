python3 test.py \
    --weights_path weights/my_model49412.h5 \
    --rnn_layer lstm \
    --use_TT \
    --data_path examples \
    --labels_path test_labels.txt \
    --lr 0.001 \
    --dropout_rate 0.25 \
    --batch_size 300 \
    --n_workers 4