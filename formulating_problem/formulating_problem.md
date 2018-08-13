Training to predict
We have some target variable y, which we want to explain using some feature vector x
step 1
Choose a model that relates the two, where the training data points will be used for tuning the model
so that it best captures the desired relation

Lets do a simple regression problem
f(x i ) = w T x i + b
y i = f(x i ) + ε i

f(x i ) is assumed to be a linear combination of some input data (x i), with a set of weights w and an intercept b. Our target output (y i) is noisy version of f(x i) after being summed with Gaussian noise ε i

step 2 
we create appropriate placeholders for our input and output data and Variables for our weights and intercept
so we have 
* (x i)
* (w)
* b
* actual value or y_true

step 3
once placeholders and variables are defined we are ready to create our model
in our example its simply 
y_pred = tf.matmul(w, tf.transpose(x)) + b

step 4
We need a good measure with which we can evaluate the models performance. To capture the discrepancey between our models predictions and the observed targets, we need a measuring reflecting "distance". This distance is often referred to as an objective or a loss function, we then optimize the model by finding the set of parameters that minimize it.
There is no ideal loss function, but choosing the most sutable one is often a creative task. The choice may depend on several factors and what types of mistakes we prefer to avoid and how easy is the model to optimize

step 5
Using right optimizer can help the model learn better

Appendix

MSE and Cross Entropy