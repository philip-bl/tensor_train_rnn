python3 deepsbd/test.py \
    --weights_path deepsbd/weights/my_model49412.h5 \
    --rnn_layer lstm \
    --use_TT \
    --data_path deepsbd/examples \
    --labels_path deepsbd/test_labels.txt \
    --lr 0.001 \
    --dropout_rate 0.25 \
    --batch_size 3 \
    --n_workers 4
