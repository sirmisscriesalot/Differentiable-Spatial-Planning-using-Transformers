#!/bin/bash
python Synthetic_Data_Manipulation.py --size 100000 --M 18 --P 90 --vis vistr18 --xfile trainx18 --yfile trainy18 --mode c --nthread 70
python Synthetic_Data_Manipulation.py --size 5000 --M 18 --P 90 --vis visva18 --xfile valx18 --yfile valy18 --mode c --nthread 70
python Synthetic_Data_Manipulation.py --size 5000 --M 18 --P 90 --vis viste18 --xfile testx18 --yfile testy18 --mode c --nthread 70
python Synthetic_Data_Manipulation.py --size 100000 --M 36 --P 180 --vis vistr36 --xfile /trainx36 --yfile trainy36 --mode c --nthread 70
python Synthetic_Data_Manipulation.py --size 5000 --M 36 --P 180 --vis visva36 --xfile valx36 --yfile valy36 --mode c --nthread 70
python Synthetic_Data_Manipulation.py --size 5000 --M 36 --P 180 --vis viste36 --xfile /testx36 --yfile testy36 --mode c --nthread 70
