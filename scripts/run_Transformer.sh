#!/bin/sh

# Running all the experiments including:
# Transformer + (O = 96) + MSE
# Transformer + (O = 96) + MAE
# Transformer + (O = 336) + MSE
# Transformer + (O = 336) + MAE
# The necessary params include predict_length && model && loss
# I highly recommend you using lr = 0.0005 when training Transformer

# 1. Transformer + (O = 96) + MSE
cd ..
echo "Begin: Transformer_(O=96)_MSE"

echo "Training..."
python train.py --model 'Transformer'  --predict_length 96 --loss 'MSE' --lr 0.0005
echo "Done."
wait
echo "Testing..."
python test.py --model 'Transformer'  --predict_length 96 --loss 'MSE'
echo "Done."
wait
echo "Predicting..."
python predict.py --model 'Transformer'  --predict_length 96 --loss 'MSE'
echo "Done."
wait

echo "Done: Transformer_(O=96)_MSE"

# 2. Transformer + (O = 96) + MAE
echo "Begin: Transformer_(O=96)_MAE"

echo "Training..."
python train.py --model 'Transformer'  --predict_length 96 --loss 'MAE' --lr 0.0005
echo "Done."
wait
echo "Testing..."
python test.py --model 'Transformer'  --predict_length 96 --loss 'MAE'
echo "Done."
wait
echo "Predicting..."
python predict.py --model 'Transformer'  --predict_length 96 --loss 'MAE'
echo "Done."
wait

echo "Done: Transformer_(O=96)_MAE"

# 3. Transformer + (O = 336) + MSE
echo "Begin: Transformer_(O=336)_MSE"

echo "Training..."
python train.py --model 'Transformer'  --predict_length 336 --loss 'MSE' --lr 0.0005
echo "Done."
wait
echo "Testing..."
python test.py --model 'Transformer'  --predict_length 336 --loss 'MSE'
echo "Done."
wait
echo "Predicting..."
python predict.py --model 'Transformer'  --predict_length 336
echo "Done."
wait

echo "Done: Transformer_(O=336)_MSE"

# 4. Transformer + (O = 336) + MAE
echo "Begin: Transformer_(O=336)_MAE"

echo "Training..."
python train.py --model 'Transformer'  --predict_length 336 --loss 'MAE' --lr 0.0005
echo "Done."
wait
echo "Testing..."
python test.py --model 'Transformer'  --predict_length 336 --loss 'MAE'
echo "Done."
wait
echo "Predicting..."
python predict.py --model 'Transformer'  --predict_length 336
echo "Done."
wait

echo "Done: Transformer_(O=336)_MAE"