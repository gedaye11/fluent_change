source common_zh_clearV4.ini

python ../compute_val_loss.py --train_collection $train_collection --val_collection $val_collection --model_name $model_name --vf_name $vf --language $lang --overwrite $overwrite


