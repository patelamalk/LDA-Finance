cd ../
mkdir ./data
mv ./Text_Filings.zip ./data
cd ./data
echo " 'Data'  dir created "
unzip Text_Filings.zip
echo " Unzipping files "
mv *.zip ../
