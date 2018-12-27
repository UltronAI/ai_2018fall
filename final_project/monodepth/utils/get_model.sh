model_name=$1
output_location=$2

filename=$model_name.zip
mkdir -p $output_location/$model_name
save_path=$output_location/$model_name

url=http://visual.cs.ucl.ac.uk/pubs/monoDepth/models/$filename

output_file=$save_path/$filename

echo "Downloading $model_name"
wget -nc $url -O $output_file
unzip $output_file -d $save_path
rm $output_file
