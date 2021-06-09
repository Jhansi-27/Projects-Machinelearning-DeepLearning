<h2 align="center">Design and impliment MultiLayerPercepetron(MLP) from scratch using Python</h2>
<h3>Description of the project:</h3>
<ul>
<li>For this assignment we are given a lander game which will randomly generate a landing zone and unsafe terrain.</li>
<li>We are required to safely land the ship without touching the unsafe terrain. Following the completion of this task, 
data will be generated automatically.</li>
<li>Using this data we need to train our own neural network with the ability to complete the same task.</li>
<li>The neural network created should suffice the following conditions:</li>
<ol>
<li>The neural network should be coded using pure Python without use of any libraries except Numpy,pandas,math</li>
<li>The neural network should perform both  Feed-Forward & Backpropagation</li>
  <ul>
  <li>Inputs:  X position to target, Y position to target</li>
  <li>Outputs: NewVelocity X, NewVelocity Y</li>
  </ul>
<li>Train the neural network created (offline training ~100 epochs) with the data collected by playing lander game.</li>
<li>Test the neural network by running the most recent weights (just feedforward) to see how the lander performs and 
compute the Root Mean Squared Error.</li></ol>
</ul>
