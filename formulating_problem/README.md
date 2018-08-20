# How to develop a Deep Learning model from scratch using TF for your custom dataset
## Training to predict
Lets relate the problem mathematically, we have some target variable y, which we want to explain using some feature vector x
### step 1
Choose a model that relates the two, where the training data points will be used for tuning the model
so that it best captures the desired relation

Lets do a simple regression problem :</br>
<code>
f(x<sub>i</sub>) = w T x i + b
y i = f(x i ) + ε i
</code>
</br>
where, f(x<sub>i</sub>) is assumed to be a linear combination of some input data (x<sub>i</sub>), with a set of weights w and an intercept b. Our target output <b>y<sub>i</sub></b> is noisy version of f(x<sub>i</sub>) after being summed with Gaussian noise ε<sub>i</sub>.

step 2 
we create appropriate placeholders for our input and output data and Variables for our weights and intercept
so we have 
* (x<sub>i</sub>)
* (w)
* b
* actual value or y_true

step 3
once placeholders and variables are defined we are ready to create our model
in our example its simply</br>
<code>y_pred = tf.matmul(w, tf.transpose(x)) + b</code>

step 4
We need a good measure with which we can evaluate the models performance. To capture the discrepancey between our models predictions and the observed targets, we need a measuring reflecting "distance". This distance is often referred to as an objective or a loss function, we then optimize the model by finding the set of parameters that minimize it.
There is no ideal loss function, but choosing the most sutable one is often a creative task. The choice may depend on several factors and what types of mistakes we prefer to avoid and how easy is the model to optimize

step 5
Using right optimizer can help the model learn better

Appendix

MSE and Cross Entropy