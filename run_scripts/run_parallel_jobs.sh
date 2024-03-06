for i in {0..5}
do
    nohup python -u run.py --run_id $i  > output_$i.log &
done
