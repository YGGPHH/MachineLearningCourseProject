#!/bin/sh

# Running all the experiments including:
# pureLSTM + (O = 96) + MSE
# pureLSTM + (O = 96) + MAE
# pureLSTM + (O = 336) + MSE
# pureLSTM + (O = 336) + MAE
# The necessary params include predict_length && model && loss

# 1. pureLSTM + (O = 96) + MSE
cd ..
echo "Begin: pureLSTM_(O=96)_MSE"

echo "Training..."
python train.py --model 'pureLSTM'  --predict_length 96 --loss 'MSE'
echo "Done."
wait
echo "Testing..."
python test.py --model 'pureLSTM'  --predict_length 96 --loss 'MSE'
echo "Done."
wait
echo "Predicting..."
python predict.py --model 'pureLSTM'  --predict_length 96 --loss 'MSE'
echo "Done."
wait

echo "Done: pureLSTM_(O=96)_MSE"

# 2. pureLSTM + (O = 96) + MAE
echo "Begin: pureLSTM_(O=96)_MAE"

echo "Training..."
python train.py --model 'pureLSTM'  --predict_length 96 --loss 'MAE'
echo "Done."
wait
echo "Testing..."
python test.py --model 'pureLSTM'  --predict_length 96 --loss 'MAE'
echo "Done."
wait
echo "Predicting..."
python predict.py --model 'pureLSTM'  --predict_length 96 --loss 'MAE'
echo "Done."
wait

echo "Done: pureLSTM_(O=96)_MAE"

3. pureLSTM + (O = 336) + MSE
echo "Begin: pureLSTM_(O=336)_MSE"

echo "Training..."
python train.py --model 'pureLSTM'  --predict_length 336 --loss 'MSE'
echo "Done."
wait
echo "Testing..."
python test.py --model 'pureLSTM'  --predict_length 336 --loss 'MSE'
echo "Done."
wait
echo "Predicting..."
python predict.py --model 'pureLSTM'  --predict_length 336 --loss 'MSE'
echo "Done."
wait

echo "Done: pureLSTM_(O=336)_MSE"

# 4. pureLSTM + (O = 336) + MAE
echo "Begin: pureLSTM_(O=336)_MAE"

echo "Training..."
python train.py --model 'pureLSTM'  --predict_length 336 --loss 'MAE'
echo "Done."
wait
echo "Testing..."
python test.py --model 'pureLSTM'  --predict_length 336 --loss 'MAE'
echo "Done."
wait
echo "Predicting..."
python predict.py --model 'pureLSTM'  --predict_length 336 --loss 'MAE'
echo "Done."
wait

echo "Done: pureLSTM_(O=336)_MAE"