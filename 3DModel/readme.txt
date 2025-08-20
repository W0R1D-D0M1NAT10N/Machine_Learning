1. To convert stl to obj
	assimp export ufov4.stl ufov4.obj

2. train.py trains the 3D NN

3. predict.py predicts with the 3D NN

4. flight_data.csv contains all the data needed to train the 3D net

5. transpose.py transposes the flight_data.csv (suitable for model training) to transposed csv format (suitable for graphing)
