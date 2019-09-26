cd ~/Workspace/git_project/pytorch-deeplab-xception

echo  "1. remove run folder...\n"
rm -rf run 

echo "2. run train.py...\n"
python train.py --backbone mobilenet --lr 0.0008 --workers 4 --epochs 1 --batch-size 4 --gpu-ids 0 --checkname 4_onnx --eval-interval 1 --dataset pascal --ft 

echo  "3. copy checkpoint to scripts folder...\n"
cp run/pascal/4_onnx/experiment_0/checkpoint.pth.tar scripts

echo  "\n4 .run demo.py...\n"
python demo.py --in-path test/ --backbone mobilenet --ckpt scripts/checkpoint.pth.tar --out-path result

echo "End..."
