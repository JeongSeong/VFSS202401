original code for

[Application of deep learning technology for temporal analysis of videofluoroscopic swallowing studies](https://doi.org/10.1038/s41598-023-44802-3)

The code will be refined within February 2024

The main command for (CONV-SA) is 
```terminal
python vfssRGBsingleTrainTest_fixed.py --split_file [split_file_name.xlsx] --rgb_root [directory/path/for/rgb/frames] --frame_len 50 --modelName PEresNet3D18Attention4 --train_type _variMem --train_batch 10 --val_batch 10 --win_limit 10 --win_stride 50
```

Command for (BIDIRECTONAL) is 
```termianl
python vfssRGBsingleTrainTest_fixed.py --split_file [split_file_name.xlsx] --rgb_root [directory/path/for/rgb/frames] --frame_len 7 --modelName resNet3D --bi True --train_batch 32 --val_batch 49 --win_limit 32 --win_stride 1
```

Command for (DEFAULT) is 
```terminal
python vfssRGBsingleTrainTest_fixed.py --split_file [split_file_name.xlsx] --rgb_root [directory/path/for/rgb/frames] --frame_len 7 --modelName resNet3D --train_batch 32 --val_batch 49 --win_limit 32 --win_stride 1
```

[split_file_name.xlsx] format
|patient_id|patient_id|num_frame|label_start|label_end|split|
|:---------|:---------|:--------|:----------|:--------|----:|
|1         |1         |92       |44         |50       |train|
|2         |2         |164      |135        |142      |val  |
|3         |3         |31       |15         |22       |test |
|...       |...       |...      |...        |...      |...  |
