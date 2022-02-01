# Differentiable-Spatial-Planning-using-Transformers
PyTorch and PyTorch Lighting implementation of the Spatial Planning Transformers (SPT) from the 2021 ICML paper "Differentiable Spatial Planning using Transformers". 

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
Visualize a dataset created using ```--mode v```. The visualization tool displays the first 10 data items from a choice of our created ```.npz``` x y file pair. An example is given below- 
```py 
python Synthetic_Data_Navigation.py --mode v --xfile trainx50 --yfile trainy50
python Synthetic_Data_Manipulation.py --mode v --xfile trainx18 --yfile trainy18
```
