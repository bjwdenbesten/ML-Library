# nanoML: ML Linear-Regression library in C++
## Library functions
#### load_csv(filepath, target_col): loads a csv into a 2d vector of doubles. Specify dependent / output variables with the second parameter.
#### write_csv(filepath, data): writes data to a csv.
#### train(X, Y): where X is the design matrix, and Y is the target matrix.
#### predict(X, Y): where X is the result of train, and Y is the data to be predicted.

### Some notes: 
- This library uses various functions / properties from the Eigen linear algebra library. Download Eigen [here](https://eigen.tuxfamily.org/index.php?title=Main_Page).
- All related functions, including reading and writing to csvs live in the namespace "ml".
- Linear regression properties live in the class "linearRegression" in the namespace "ml".
- To specify gradient descent over the normalEQ, and set the LR to 0.0001 and iterations to 5000, construct a class instance as such: ml::linearRegression m(true, 0.001, 5000).
- There are two examples set up already for the normalEQ and GD. Find them in "training_examples".
- To compile the project run "cmake ..", and then "make".
