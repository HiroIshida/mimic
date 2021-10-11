SCRIPT_DIR=$(cd $(dirname $0); pwd)
python3 $SCRIPT_DIR/kuka_reaching.py -n 10
python3 -m mimic.scripts.train_auto_encoder -pn kuka_reaching -n 3
python3 -m mimic.scripts.train_lstm -pn kuka_reaching -n 10
python3 -m mimic.scripts.predict -pn kuka_reaching
