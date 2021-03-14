DATA_DIR="./data"
RUN_DIR="./run"
export PYTHONPATH="${PWD}:${PYTHONPATH}"

echo "int8 DQ"
python reddit_binary/main.py --int8 --gc_per --lr 0.005 --DQ --low 0.0 --change 0.1 --wd 0.0002 --outdir ${RUN_DIR} --path ${DATA_DIR} | tee int8_dq.txt
echo "int8 normal"
python reddit_binary/main.py --int8 --ste_mom --lr 0.005 --wd 0.0002 --epochs 200 --outdir ${RUN_DIR} --path ${DATA_DIR} | tee int8.txt
echo "int4 DQ"
python reddit_binary/main.py --int4 --gc_per --lr 0.001 --DQ --low 0.1 --change 0.1 --wd 4e-5 --epochs 200 --outdir ${RUN_DIR} --path ${DATA_DIR} | tee int4_dq.txt
echo "int4 normal"
python reddit_binary/main.py --int4 --ste_mom --lr 0.05 --epochs 200 --outdir ${RUN_DIR} --path ${DATA_DIR} | tee int4.txt

