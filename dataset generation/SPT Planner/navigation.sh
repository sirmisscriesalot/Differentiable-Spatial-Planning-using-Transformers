#!/bin/bash
python Synthetic_Data_Navigation.py --size 100000 --M 15 --xfile trainx15 --yfile trainy15 --mode c --nthread 80
python Synthetic_Data_Navigation.py --size 5000 --M 15 --xfile valx15 --yfile valy15 --mode c --nthread 80
python Synthetic_Data_Navigation.py --size 5000 --M 15 --xfile testx15 --yfile testy15 --mode c --nthread 80
python Synthetic_Data_Navigation.py --size 100000 --M 30 --xfile trainx30 --yfile trainy30 --mode c --nthread 80
python Synthetic_Data_Navigation.py --size 5000 --M 30 --xfile valx30 --yfile valy30 --mode c --nthread 80
python Synthetic_Data_Navigation.py --size 5000 --M 30 --xfile testx30 --yfile testy30 --mode c --nthread 80
python Synthetic_Data_Navigation.py --size 100000 --M 50 --xfile trainx50 --yfile trainy50 --mode c --nthread 80
python Synthetic_Data_Navigation.py --size 5000 --M 50 --xfile valx50 --yfile valy50 --mode c --nthread 80
python Synthetic_Data_Navigation.py --size 5000 --M 15 --xfile testx50 --yfile testy50 --mode c --nthread 80