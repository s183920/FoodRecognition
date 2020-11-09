#!/bin/bash
# curDir="{$PWD | rev | cut -d'/' -f-2 | rev}"
# echo $curDir

# if [ $curDir = FoodRecognition/data ]
if true
then
	if [ -d "VOC" ] 
	then
		echo -e "\e[32mData already downloaded!\e[0m"
	else
		wget "https://drive.google.com/u/0/uc?id=1hz5AxTTVcfQz1dK0uoyfE89fMrZXQEzk&export=download"
		mkdir VOC
		unzip VOC.zip -d VOC

		mkdir VOC/VOCdevkit/VOC2007
		unzip -o VOC/VOCdevkit/VOC2007.zip "VOC/VOCdevkit/VOC2007/*"
		
		###If the downloaded VCTK is in tar.gz, run this:
		#tar -xzvf VCTK-Corpus.tar.gz -C ../data/VCTK-Data
	fi
	
else
	echo -e "\e[31mHave you made sure to run this scripts from the 'FoodRecognition/data' directory?\e[0m"
fi
