# Malaria-Detection-Using-ML
Machine Learning based model to detect Malaria using cell images using traditional Image Processing Techniques. 

## Libraries Used:
  1.Scikit Learn
  2.Pandas
  3. Scikit Image
  4.OpenCV
  
## Dataset:
Dataset was taken from - [Malaria Dataset](https://ceb.nlm.nih.gov/repositories/malaria-datasets/)
It consists of 27558 images (Parasitized - 13779, Uninfected- 13779)

## Feature Extraction:
Features extracted from these images include:
 1. 13 GLCM based features obtained by performing Morphological operations on input images. [Malaria_detection_using_morphological_tech.csv]
 2. 13 GLCM based features obtained by performing Sobel filtering on input images. [Malaria_with_Sobel_New.csv]
 3. 9 features extracted by applying Gabor filters(2), Local Binary Patterns(2), and GLCM (5) on Grayscale input images.[DATASET2.csv]
 4. 9 features obtained by applying Gabor, LBP and GLCM on denoised input images (Non-Local means filtering).[DATASET.csv]

These features were extracted using python and MATLAB

## Algorithms Used:
KNN, SVM and RandomForest algorithms are used to create models on the given features extracted above.
 
### Accuracies:
  #### Malaria Detection using morphological techniques: 
    1.KNN - 88.66
    2.SVM - 91.76
    3.RF -  90.18  
  #### Malaria Detection with Sobel Filtering: 
    1.KNN - 72.09
    2.SVM - 72.07
    3.RF -  71.9   
  #### Malaria Detection using Gabor, LBP and GLCM: 
    1.KNN - 77.12 
    2.SVM - 86.84
    3.RF -  80.4  
 #### Malaria Detection using Gabor, LBP and GLCM on denoised images: 
    1.KNN - 74.16
    2.SVM - 86.52
    3.RF -  76.9
The highest accuracy of 91.76% was obtained from the SVM Model on the GLCM based Dataset

## Comparison with CNN:
A basic CNN model was deployed on the dataset with an accuracy of 93.75%. It can be observed that traditional ML models were as good as the CNN model. 
