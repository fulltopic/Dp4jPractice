for i in *.jpg
	do convert $i -colorspace gray /home/zf/books/DeepLearning-master/majiong/dataset4/$i
done
