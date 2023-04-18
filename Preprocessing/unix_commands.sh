## Run the command in the trial directory
trial=11
trialdir="trials/${trial}"
echo $trialdir

# Loop through the excel files and preprocess them to output.txt directory
for i in $(seq 1 9)
do
  cat "Trial${trial}Zebrabox_raw_000${i}.xls" | awk '{print $1 "\t" $3 "\t" $4 "\t" $6 "\t" $7}' | grep  -v "\t0\t0" | perl -ne '$line = $_; @ele = split(/\t/, $line); if ($ele[1] == 108) {print $line;}' > "output${i}.txt"
done
