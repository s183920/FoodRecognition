#!/bin/bash

if [ -f  VOC.zip ]
	then
		# original link: https://drive.google.com/file/d/1hz5AxTTVcfQz1dK0uoyfE89fMrZXQEzk/view
		unzip VOC.zip
		rm VOC.zip

		unzip VOC/VOCdevkit/VOC2007.zip -d VOC/VOCdevkit/
		rm VOC/VOCdevkit/VOC2007.zip
		

elif [-d VOC ]
	then
		echo -e "\e[32mData already unzipped!\e[0m"

else
	# echo -e "\e[31mHave you made sure to run this scripts from the 'FoodRecognition/data' directory?\e[0m"
	echo -e "\e[31mPlease run this script from the folder containing VOC.zip"
fi
