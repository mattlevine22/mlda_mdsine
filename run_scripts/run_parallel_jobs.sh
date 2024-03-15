for i in {0..1}
do
    nohup python -u run.py --run_id $i --run_all 0 --project_name mdsine2_v5  > output_mech_then_mark$i.log &
    echo $! > save_pid_mech_then_mark$i.txt
done
