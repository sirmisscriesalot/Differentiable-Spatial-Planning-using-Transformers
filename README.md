# Differentiable-Spatial-Planning-using-Transformers
PyTorch and PyTorch Lightning implementation of the Spatial Planning Transformers (SPT) from the 2021 ICML paper "Differentiable Spatial Planning using Transformers". 

To generate the synthetic datasets for navigation planner with 100000/5000/5000 train validation test split for 15X15, 30X30 and 50X50 map sizes- 
```
cd dataset\ generation/SPT\ Planner/
bash navigation.sh
```
To generate the synthetic datasets for manipulation planner with 100000/5000/5000 train validation test split for 18X18 and 36X36 map sizes - 

```
cd dataset\ generation/SPT\ Planner/
bash manipulation.sh
```
You can also create your own custom datasets using the Synthetic_Data_Navigation.py and Synthetic_Data_Manipulation.py python scripts. An example is given below -
```py
python Synthetic_Data_Navigation.py --size 10000 --M 100 --xfile trainx100 --yfile trainy100 --mode c --nthread 80
python Synthetic_Data_Manipulation.py --size 10000 --M 72 --P 90 --vis vistr72 --xfile trainx72 --yfile trainy72 --mode c --nthread 70
```
For more detailed information on what each optional argument does use - 
```py
python Synthetic_Data_Navigation.py --help
python Synthetic_Data_Manipulation.py --help 
```
Visualize a dataset created using ```--mode v```. The visualization tool displays the first 10 data items from a choice of our created ```.npz``` x y file pair which consists of matrixes m,g and the ground truth for the navigation dataset. Visualizi

ng the manipulation dataset uses a different pickle file separately. An example on how to use the tool is given below- 
```py 
python Synthetic_Data_Navigation.py --mode v --xfile trainx50 --yfile trainy50 --M 50
python Synthetic_Data_Manipulation.py --mode v ---vis vistr36 --M 18
```
![alt text](https://github.com/sirmisscriesalot/Differentiable-Spatial-Planning-using-Transformers/blob/main/dataset%20generation/SPT%20Planner/manipulation_test_data.png?raw=true ) 

A sample Manipulation Data Item

![alt text](https://github.com/sirmisscriesalot/Differentiable-Spatial-Planning-using-Transformers/blob/main/dataset%20generation/SPT%20Planner/navigation_test_data.png?raw=true )

A sample Navigation Data Item

Find the model we used to train [here](https://colab.research.google.com/drive/1n_e0a452BACcg5p2TeJwSRpFd0aKLUK6) on google colab. You can also find it in ```utils/dspt_planner.py```. Manually specify the file location of the custom datasets and other parameters in the config dictionary if you wish to train the model locally or on custom datasets. 

![alt text](https://github.com/sirmisscriesalot/Differentiable-Spatial-Planning-using-Transformers/blob/main/utils/predicted_output.jpg)

A sample predicted output vs ground truth map



https://user-images.githubusercontent.com/19360845/152526089-8388faf2-0b08-4569-bdc6-1670d80d1b78.mp4

