# Fairly Diabetic Retinopathy Detection from Color Fundus Photos utilizing EfficientNet and ResNet models

## Dataset settings

<img width="290" alt="image" src="https://github.com/user-attachments/assets/1b4ce5c9-997c-4294-9ada-e5695b180685">

Files in each folders are in `.npz` formate which contains keys and values, i.e.

```
Keys in the .npz file: ['slo_fundus', 'race', 'male', 'hispanic', 'maritalstatus', 'language', 'dr_class', 'dr_subtype']
Race: 2
Male: 0
Hispanic: 0
Marital Status: 1
Language: 0
Diabetic Retinopathy Class: 0
Diabetic Retinopathy Subtype: no.dr.diagnosis
```

Here, ``dr_class`` represents class whether it has ``DR(1) or not(0)``.

![image](https://github.com/user-attachments/assets/d2736a06-e727-40cd-9fa3-e60f2e0a7054)

```
print('Train data: ',len(train_data),'Train image[0] shape: ', np.asarray(train_data).shape)
print('Valid data: ',len(valid_data),'Validation image[0] shape: ', np.asarray(valid_data).shape)
print('Test data: ',len(test_data),'Test image[0] shape: ', np.asarray(test_data).shape)
```
shows shapes of the data as,

```
Train data:  4476 Train image[0] shape:  (4476, 200, 200, 3)
Valid data:  641 Validation image[0] shape:  (641, 200, 200, 3)
Test data:  1914 Test image[0] shape:  (1914, 200, 200, 3)
```

The demographic distributions found as,

```
Race Distribution: Counter({1: 1545, 0: 1468, 2: 1463})
Male Distribution: Counter({1: 2390, 0: 2086})
Hispanic Distribution: Counter({0: 2247, 1: 2229})
Language Distribution: Counter({1: 1530, 2: 1499, 0: 1447})
Maritalstatus Distribution: Counter({3: 928, 4: 912, 1: 907, 2: 889, 0: 840})
```

## Models

The EfficientNet and ResNet model designed simply as,

```
self.base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3)
            )
```
and
```
self.base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3)
            )
```

## Fairness employing

1. Metrics like male_recall, female_recall, and fairness_penalty are employed during training. These metrics ensure that the model's performance is monitored separately for male and female groups, addressing potential fairness concerns.

2. Fairness_penalty is a calculated value likely aiming to reduce disparities between groups, such as ensuring similar recall rates for males and females.

### Fairness Aware Training

Fairness is incorporated into the training process through a custom fairness penalty term added to the loss function. This penalty quantifies performance disparities between male and female subgroups. For each batch during training, the recall for both male and female groups is calculated, and the fairness penalty is defined as the absolute difference between these recall values.

The modified loss function is expressed as:

<img width="580" alt="image" src="https://github.com/user-attachments/assets/873debc3-2c18-406b-aeb5-89377e58ff4b">


## Train EfficientNet

To train this model you need to run,

```
print("Training EfficientNet model...")
efficientnet_model, efficientnet_history, efficientnet_results = train_model('efficientnet')
```

## Train ResNet

To train this model you need to run,

```
print("\nTraining ResNet model...")
resnet_model, resnet_history, resnet_results = train_model('resnet')
```
