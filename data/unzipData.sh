#!/bin/bash

if [ -f  VOC.zip ]
	then
		# original link: https://drive.google.com/file/d/1hz5AxTTVcfQz1dK0uoyfE89fMrZXQEzk/view
		# for VOC2007 zip file: https://drive.google.com/file/d/1xQ3PmHLRKbhgcpF2D2dv0xNr1bjIFaCB/view
		sudo add-apt-repository universe
		sudo apt update	
		sudo apt install p7zip-full p7zip-rar

		# unzip VOC.zip
		# rm VOC.zip

		cd VOC/VOCdevkit
		7z x VOC2007.7z
		cd ../..
		# rm VOC/VOCdevkit/VOC2007.zip

		
		

elif [-d VOC ]
	then
		echo -e "\e[32mData already unzipped!\e[0m"

else
	# echo -e "\e[31mHave you made sure to run this scripts from the 'FoodRecognition/data' directory?\e[0m"
	echo -e "\e[31mPlease run this script from the folder containing VOC.zip\e[0m"
fi
