#!/bin/bash
ls *.png | shuf | while read line; do
	python -W ignore batchSegment.py --minLabels 20 --input $line --visualize 0 --nConv 4 --nChannel 100 --maxIter 1 --lr 0.02 --output output/output_$line
done
