Math modelling exam paper

**Evidence outline:**

* Exploratory data analysis  
* Baseline model  
* Improved model  
* Generalization and analysis  
* Temporal prediction impact


**Topics outline:**  
**Introduction:**

* Background for the problem.  
  * In Copenhagen, there are 380km of bicycle lanes and 40% of the commuters commute by bicycle, with a population of \>600k residents. In such a city, the ability to predict the bike traffic could help with e.g. real-time traffic management, or with estimating CO2 savings from cycling.   
  * The work attempts to perform near future \- 1 week forward \- bike traffic prediction. We have fitted and analysed a ridge regression on 77 **(fix to the original number)** raw and engineered calendar, weather and temporal features.   
  * The goal of the report is to evaluate the feasibility of this model. 
  Purpose: investigate whether a regularised linear model on weather + calendar features is suitable for predicting near-future bicycle traffic in Copenhagen. 
* What is the goal?  
  * Try to accurately predict bicycle traffic in the near future.  
* What will be done?  
  * We do:  
    * Ridge regression   
    * Feature engineering

Yeah, just say what you are going to do, and that this model is going to be try.  
**Add a table of contents**  
Add it\!  
**Data**  
Describe the data as it was  
**Methods**

* Describe ridge regression per ISLP book. Seriously, don’t wanna go inside.  
* Motivate the use of regression **and discuss the assumptions of the model.**  
  * It avoids overfitting in high-dimensional settings (due to the reduction of the model’s flexibility). It does so by pushing all coefficients towards the zero and so reducing the flexibility of the model.  
  * It helps to avoid issues stemming from the multicollinearity (due to avoidance of exploding coefficients)  
  * The assumption that multiple features contribute roughly evenly. It could be the weather.  
  * OK, for now I will keep it as an limitation, and will not got much in depth. Just let me write about the lasso.  
  * OK, let me it be. What is the problem to include it?  
  * Lasso: assume all coefficients are zero. Ridge \- features contribute slightly.  
  * From my perspective, I believe that only a few matter, but will see how it goes. It is hard to estimate upfront, honestly speaking. OK, something like this. Say that ridge assumes that coefficients are normally distributed.  
  * **Yeah, i need to discuss if these are problems or not.** If this is a problem \- argue why; if this is not a problem \- again, discuss if yes/no. That would be the best.  
  * Anything to write about it a plus, then i don’t need to write more about the others.
  * Purpose: investigate whether a regularised linear model on weather + calendar features is suitable for predicting near-future bicycle traffic in Copenhagen.  
* Scaling before doing any regression  
* Feature engineering  
  * Just write in general which we obtained. In general, that is the best.  
  * Cyclic features  
  * Polyjomial features (interactions and squares)  
  * Indicator variables :)  
  * Mention the extension here in my humble opinion.  
  * *Write about the extension 5b*  
* Explain how the splits were made/evaluation is made.  
* Handling of missing data \- using means, and explain why should it actually work? Nope, let’s leave it for the later? Yeah, it could be? No. Let’s write about using mean (due to single missing points \+ use of the temporal model later).  
* Citation \- using sklearn library.  
* **Also for that \- the code should be at least, minimally cleaned \- though I am pretty sure, they were not going to run it.** So simply add explanations.
* Metrics

**Experiments and Results**

* Write that we use Use Dronning Lousises Bro actually, because it was the biggest station.  
* First \- fit only linear regression; for one station only. Provide MSe  
  * Discuss what could be missing AND motivate the use of the regression algorithms 
    * Weekends (visualise that)  
    * Cyclical problems (visualise that).  
    * Polynomial features / interactions \- not sure how to realise that. These can be added, because they allow for exploration of better features with low risk of overfitting. Ridge and Lasso regressions prevent that.   **Now it is clear: display a residual plot from the independent variable and say what it could actually mean**
    * 2. The narrative bridge from baseline to improved model is missing. This is exactly the thing we worked through in chat and it isn't yet in the outline. The chain should be: EDA reveals patterns → OLS baseline captures the easy ones → residual diagnostics show specific failures (sinusoidal residuals by hour, group-dependent residuals by weekday/weekend, monotonic-non-zero residuals vs temperature) → each engineered feature class is the fix for one diagnosed failure → expansion creates multicollinearity → ridge resolves that via the eigenvalue argument from the project PDF. Right now the outline says "polynomial features / interactions — not sure how to realise that." That sentence is the report's weakest link; replacing it with the residual-driven story is probably the single highest-leverage edit you can make.
    * 3. The ridge motivation doesn't yet exploit the course material. The exam PDF spends a full page on the eigenvalue / Tikhonov derivation — that's a reviewer hint that they want to see you use it. Your outline mentions multicollinearity in passing but doesn't reference the inequality E(L²) > σ²/λ_p or the eigenvalue-shift mechanic. Two concrete additions:
    * A one-sentence reference to the bound from the PDF, framed as "the expanded feature set produces correlated columns by construction (e.g. temp with temp², cyclic features with their interaction products), which makes the smallest eigenvalues of XᵀX small and inflates the OLS variance per the bound in §3 of the project text."
    * An empirical hook: report the condition number (or the ratio λ_max/λ_min) of XᵀX for your engineered feature matrix. One number, one sentence, and your "ridge is needed" claim becomes verifiable instead of asserted.
* Fitting ridge and lasso \- again, on the same model. DO BELOW FOR BOTH\!\!\!\!  
  * Write specifically, which features were used.  
  * Write which k was picked.  
  * Provide MSE  
  * Difference in MSE between both.  
  * Visualisation of the coefficients top 5 and low 5 coefficients.  
    * Top 5 \- makes a lot of sense. (temperature, wind etc.)  
    * Low 5 \- write about them.    
  * Generalization and Analysis  
  * I propose \- you go with Ridge, because apparently lots of features truly matter. Which is remarkable.  
* Generatlisation  
  * Single validation:  
    * Report the number   
    * Compare with the validation (it was good, as it was super similar)  
  * Train one model for all stations and evaluate on all the other stations \- have a look at how this performs.  
  * See the transferabilit yand how it workscowo

**Discussion**  
• Principles, relations, and general conditions shown in results  
	While the temperature is quite important it was not the only one.  
	It seems that the bike traffic is a function of many variables instead of a few ones.  
• Stress exceptions  
	Sure, could have been other models.  
• Relate to what was expected  
	Again, surprising that multiple features actually have contrbitued  
• Discuss usefulness of the analysis/mode  
	Not sure? Yeah. I am not sure about what to write here.
  5. Discussion is thin. The Salami feedback explicitly called out reports where the discussion is "3-4 lines" or restates numbers without interpretation. Your current discussion has three half-sentences. Things to add:

Why ridge specifically over lasso (your lasso.txt shows essentially identical RMSE; that's a discussion point — "regularisation type matters less than the regularisation amount in this regime, because the surface is flat over a wide λ band").
Why a linear model at all: acknowledge that a tree-based model (XGBoost) or a Poisson GLM would likely outperform — you're not doing them because the project is constrained to linear regularised models, not because they're inferior.
Why the spikes fail: bike traffic spikes are driven by unobserved events (concerts, demonstrations, festivals, weather warnings) that are not in any of your features. No amount of feature engineering on (temperature, wind, precipitation, calendar) will capture them. This is a data limitation, not a model limitation.
Data representativity: 2021 was COVID-affected — commute patterns were unusual. Briefly note this as a caveat for generalisation.

* Well, i guess gonna be easier to write once I have everything. Yeah.

**Conclusion**

* It is sufficiently accurate or it is not \- make it a statement. From my perspective: it is not sufficient, due to not being able to handle spikes quite drastically.  
  * I would say \- there is clearly a potential, however it is clearly unable to handle the spikes. I think it is fine.

