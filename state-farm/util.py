import os
import pandas as pd

base_dir = '/home/lifesailor/.kaggle/state-farm/'
train_dir = '/home/lifesailor/.kaggle/state-farm/imgs/train/'


print("READ CSV...")
df = pd.read_csv(base_dir + 'driver_imgs_list.csv')

for index, row in df.iterrows():
    path = os.path.join(train_dir, row['classname'])
    os.rename(os.path.join(path, row['img']), os.path.join(path, row['subject'] + "_" +
                                                                 row['classname'] + "_" +
                                                                 str(index) + '.jpg'))

