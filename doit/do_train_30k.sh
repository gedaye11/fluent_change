source common_zh3.ini
#python ../generate_vocab.py $train_collection --language $lang --rootpath $rootpath
python ../trainer.py --model_name $model_name  --train_collection $train_collection --language $lang --vf_name $vf --overwrite $overwrite --fluency_method $fluency_method

