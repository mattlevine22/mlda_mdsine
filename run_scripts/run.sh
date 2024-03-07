for i in {0..3}
do
    nohup python -u run.py --run_id $i  > output_$i.log &
done
