#!/bin/bash
ls *.png | shuf | while read line; do
        echo "
	module load slideLearn
	python batchSegment.py --minLabels 3 --input $line --visualize 0 --nConv 2 --nChannel 100 --maxIter 500 --lr 0.05 --output output/output_$line" | qsub -N $line -cwd -e errors/ -o logs/ -l h_rt=100G -l h_rt=1:30:00
done
