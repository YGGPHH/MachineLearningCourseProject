#!/bin/sh

# Running all the experiments including:
# TimeSeriesCNN + (O = 96) + MSE
# TimeSeriesCNN + (O = 96) + MAE
# TimeSeriesCNN + (O = 336) + MSE
# TimeSeriesCNN + (O = 336) + MAE
# The necessary params include predict_length && model && loss

# 1. TimeSeriesCNN + (O = 96) + MSE
cd ..
echo "Begin: TimeSeriesCNN_(O=96)_MSE"

echo "Training..."
python train.py --model 'TimeSeriesCNN'  --predict_length 96 --loss 'MSE'
echo "Done."
wait
echo "Testing..."
python test.py --model 'TimeSeriesCNN'  --predict_length 96 --loss 'MSE'
echo "Done."
wait
echo "Predicting..."
python predict.py --model 'TimeSeriesCNN'  --predict_length 96
echo "Done."
wait

echo "Done: TimeSeriesCNN(O=96)_MSE"

# 2. TimeSeriesCNN + (O = 96) + MAE
echo "Begin: TimeSeriesCNN_(O=96)_MAE"

echo "Training..."
python train.py --model 'TimeSeriesCNN'  --predict_length 96 --loss 'MAE'
echo "Done."
wait
echo "Testing..."
python test.py --model 'TimeSeriesCNN'  --predict_length 96 --loss 'MAE'
echo "Done."
wait
echo "Predicting..."
python predict.py --model 'TimeSeriesCNN'  --predict_length 96
echo "Done."
wait

echo "Done: TimeSeriesCNN_(O=96)_MAE"

# 3. TimeSeriesCNN + (O = 336) + MSE
echo "Begin: TimeSeriesCNN_(O=336)_MSE"

echo "Training..."
python train.py --model 'TimeSeriesCNN'  --predict_length 336 --loss 'MSE'
echo "Done."
wait
echo "Testing..."
python test.py --model 'TimeSeriesCNN'  --predict_length 336 --loss 'MSE'
echo "Done."
wait
echo "Predicting..."
python predict.py --model 'TimeSeriesCNN'  --predict_length 336
echo "Done."
wait

echo "Done: TimeSeriesCNN_(O=336)_MSE"

# 4. TimeSeriesCNN + (O = 336) + MAE
echo "Begin: TimeSeriesCNN_(O=336)_MAE"

echo "Training..."
python train.py --model 'TimeSeriesCNN'  --predict_length 336 --loss 'MAE'
echo "Done."
wait
echo "Testing..."
python test.py --model 'TimeSeriesCNN'  --predict_length 336 --loss 'MAE'
echo "Done."
wait
echo "Predicting..."
python predict.py --model 'TimeSeriesCNN'  --predict_length 336
echo "Done."
wait

echo "Done: TimeSeriesCNN_(O=336)_MAE"
