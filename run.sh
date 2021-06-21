# bass-ce
init_epochs=200
init_batch_size=128
init_learning_rate=0.1

stu_epochs=10
stu_batch_size=256
stu_learning_rate=0.05

ce=True
ce_ind=False
balance=False

python3 data_process_with_agument.py --iter 0
python3 main_supcon.py --temp 0.1 --cosine --mode train --iter 0 --epochs ${init_epochs} --batch_size ${init_batch_size} --learning_rate ${init_learning_rate}
python3 main_supcon.py --temp 0.1 --cosine --mode test --iter 0 --epochs ${init_epochs} --batch_size ${init_batch_size} --learning_rate ${init_learning_rate}
python3 main_supcon.py --temp 0.1 --cosine --mode pseudo --iter 0 --epochs ${init_epochs} --batch_size ${init_batch_size} --learning_rate ${init_learning_rate}
#
file_path=./save/SupCon/
# Prediction jobs for different shards can run in parallel if you have multiple GPUs/TPUs
for iter in {1..2}
do
	python3 data_process_with_agument.py --file_path ${file_path} --iter ${iter} --balance ${balance}
	python3 main_supcon.py --temp 0.1 --cosine --mode train --iter ${iter} --epochs ${stu_epochs} --batch_size ${stu_batch_size} --learning_rate ${stu_learning_rate}
	python3 main_supcon.py --temp 0.1 --cosine --mode test --iter ${iter} --epochs ${stu_epochs} --batch_size ${stu_batch_size} --learning_rate ${stu_learning_rate}
	python3 main_supcon.py --temp 0.1 --cosine --mode pseudo --iter ${iter} --epochs ${stu_epochs} --batch_size ${stu_batch_size} --learning_rate ${stu_learning_rate}
done
