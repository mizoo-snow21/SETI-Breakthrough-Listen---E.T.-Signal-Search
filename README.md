# SETI-Breakthrough-Listen---E.T.-Signal-Search

https://www.kaggle.com/c/seti-breakthrough-listen/leaderboard  
We got 92th prize.  
This is summary and codes.

# Summary

## Data strategy
・5fold StratifiedKFold  
・Using new data  

# Model
## efficientnet_b0_ns
・img_size = 512 x 512  
・optimizer = sam  
・epoch = 20  
・scheduler = GradualWarmupScheduler + CosineAnnealingLR  

## efficientnet_b3_ns
・img_size = 512 x 512  
・optimizer = AdamP  
・epoch = 17    
・scheduler = GradualWarmupScheduler + CosineAnnealingLR  

## eca-nfnetl0  
・img_size = 512 x 512  
・optimizer = Ranger  
・epoch = 15    
・scheduler = ReduceLROnPlateau  

## efficientnet_v2_s    
・img_size = 512 x 512  
・optimizer = AdamP  
・epoch = 15    
・scheduler = GradualWarmupScheduler + CosineAnnealingLR 

## Weighte optimazation  
 cv = 0.8824  
 lb = 0.77510  

# Some Settings
## Augmentaion
```
def get_transforms(*, data):
    
    if data == 'train':
        return A.Compose([
            A.Resize(CFG.size, CFG.size, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(rotate_limit=0, p=0.25),
            A.OneOf([
                A.MotionBlur(p=1.0),
                A.GaussianBlur(p=1.0),
                A.GaussNoise(p=1.0),
            ], p=0.2),
            A.OneOf([
                A.OpticalDistortion(distort_limit=1.0, p=1.0),
                A.GridDistortion(num_steps=5, distort_limit=1.0, p=1.0),
                A.ElasticTransform(alpha=3, p=1.0),
            ], p=0.2),
            A.IAASharpen(p=0.25),
            A.Cutout(p=0.3),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return A.Compose([
            A.Resize(CFG.size, CFG.size),
            ToTensorV2(),
        ])
```
・mixup(alpha=0.5)  

## Loss
BCEWithlogitsloss(pos_weight=1.5)  


# Not Worked
・Loss function(FocalCosineLoss/CrossEntropyLoss)  
・LRscheduler(OneCycleLR/LambdaLR)  
・Additional data(old)  
・cutmix  
