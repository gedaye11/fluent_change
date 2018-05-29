source common_zh_weiboV2.ini
python ../generate_vocab.py $train_collection --language $lang --rootpath $rootpath
python ../trainer.py --model_name $model_name  --train_collection $train_collection --language $lang --vf_name $vf --overwrite $overwrite

