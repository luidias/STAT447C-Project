"Project.R"
This is the full R script for the project, including analysis and custom prediction functions.

"ProjectFull.Stan"
The main stan file. Includes optional pseudo-LOO code within the stan framework.

"ProjectFullWithInference.Stan"
Similar to ProjectFull but takes an additional set of variables as input, over which it infers the response. No pseudo-LOO code.

"Loadeddata.Rdata"
R data package with the contents of the data folder. Loading this into the environment is faster than re-running the code to load it within Project.R.

"FitFull1000itersWithPseudoLOO.RData"
The output of a 1000-iteration, 1-chain Stan simulation of my full dataset, including the pseudo-LOO predictions. This takes several hours to generate, so re-use when possible unless you need to modify the Stan model or simulation parameters.
Update: this was too large for an upload to git, unfortunately.