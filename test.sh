language_list=('all')
test_list=('de')
export CUDA_VISIBLE_DEVICES='0,1'
for lg in ${language_list[*]}; do
for test_lg in ${test_list[*]}; do
    python3 predict.py \
    --lg $lg \
    --test_lg $test_lg \
    --layer_id 12\
    --dataset_path ./data/ \
    --queue_length 0 \
    --load_model_path ./model/pytorch_model.bin \
    --unsupervised 0 
done
done
# test_list=('de' 'es' 'fr' 'ru' 'ko' 'ja' 'zh' 'ar')