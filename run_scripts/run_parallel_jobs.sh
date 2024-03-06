for i in {0..3}
do
    nohup python -u run.py --run_id $i --run_all 0 --project_name mdsine2_v3  > output_$i.log &
    echo $! > save_pid_$i.txt
done
