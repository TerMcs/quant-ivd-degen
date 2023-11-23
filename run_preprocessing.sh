for i in 1 2 4 8 16 32 64
do
	for j in "none" "0270" "0135"
	do 
		dvc exp run -S data.binwidth=$i -S data.resampling=$j
	done
done