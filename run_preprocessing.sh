for i in 1 2 4 8 16 32 64
do
	for j in "none" "0270" "0135"
	do 
		dvc exp run -S prepare.binwidth=$i -S prepare.resampling=$j
	done
done