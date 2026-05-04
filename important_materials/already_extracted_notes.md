### Report (in general)
73
Coherent report which independently describes the analysis
– Introduction with problem statement
– Description of method – the mathematical model (general)
– Description of experiment, analysis, and results (concrete)
– Discussion of results:
▪ Does the method solve the problem?
▪ How good is the solution?
▪ What parameters have been used?
▪ Are the data representative of the problem?
▪ What are the limitations of the method?
– Conclusion – what is the contribution of the analysi

### Introduction
Introduction: 4 components
probably (not necessarily) 4 paragraphs
1. Context (provide context to readers, establish importance of work)
2. Need (state the need of the work, what the community currently has and what it wants)
3. Task (what you have done to address the need)
4. Object (preview the remainder of the report to mentally prepare reader

Introduction
• Describes the problem – why bother with the investigation/modelling
• Describes relevance – who benefits from work (ask why?)
– “We wish to suggest a structure for the salt of deoxyribose nucleic acid
(D.N.A.). This structure has novel features which are of considerable biological
interest.” Crick & Watson, Nature, 1953
• Describes related work – how has others approached the task (ask how?) (but here we skip id)
• Argues for the choice of method


### Methods and materials
• Describes the method/model in details (the mathematics are described)
• The description should be detailed enough for others to reproduce the experiments
• Reference to known methods – no need to describe everything
• Results should not be described here
• The method description is general
– Materials are not

• Method sections are boring to read, descriptive
– yet they need not be
• Make this part interesting:
– explain your choices
– what is special or different in your approach, why?
– move detailed, less important parts to the end of the report (Appendix)

### Experiments and Results
• Describes data, experiments and results
• Others should be able to reproduce the experiments
• Data and results are described concise
– Use statistics – mean values, standard deviations, confidence intervals etc.
– Use representative examples
– Point out general trends
– Point out deviations, surprises and failure cases
• Method should not be described here
• Experiments and results are specifi


### Discussion
Discussion
• Principles, relations, and general conditions shown in results
• Stress exceptions
• Relate to what was expected
• Discuss usefulness of the analysis/model
• (Future work / extensions)
Results and Discussion are often combined because results make little sense
to most readers without interpretation

### Conclusion
• Most important outcome of your work
• What are the main results
• Contribution with concise argumentation: What do we know now, that we did not know before
• Not a summary
• Do not restate what you have done or what the paper does.
• Instead, focus on what you have found and, especially, on what your findings mean

## From lecture 5 (feedback from optical flow 1)
Introduction (Discussion & Conclusion)
1. Entice the reader
2. Onset for the ”story telling”
– Data, Discussion and Conclusion
• General remarks:
– Lack of purpose
– Strong indications of it being an exercise

Introduction (next report)
5
End* your introduction with a bullet point* stating:
• What is the purpose/goal of the report?
Concrete exercise for the next report:
• Introduction/purpose is not fixed, although the data is!
*It doesn’t have to be the actual end nor an actual bullet point
Example 2:
– Purpose: ~ compare the potential of *M1(simple)*
vs. *M2 (advanced)* for the application/problem
– Discussion: Compare “accuracy”,
“generalizability”, “scalability” and “tunability” etc.
– Conclusion: *M1* or *M2* or “depends on x”
Example 1:
– Purpose: ~ investigate if *Mx* is suitable for
the application/problem
– Discussion: Sufficiently “accurate”,
“generalizable”, “scalable” and “tunable” etc.?
– Conclusion: “No!” or “probably not” or
“potentially” or “Yes!”

• Linking: Introduction to Discussion to Conclusion
– Ensure a goal for the report*



### Ideas for me:
* Simpler model needs a clear purpose: why has it been done at all? 
One way is to argue that:
    * it was necessary to quickly assess which features were missing? (I think so, it's the best argumentation for that? how would i motivate using all the features? or should i perhaps simplify it and cut it down?)
    * it was done as a comparison between simpler and a stronger model?
    * How to construct the introduction then?
    Let's commit to this first: investigate if the model is suitable for predicting near-future bike traffic.
    And then I will use the baseline to guide the features. Again: what is the argument to use ridge regression at all then?
    It makes sense to use it if there are many features, and/or if there is multicollinearity happening. 
    So, I would either need to argue for use of many features or, by the fact there was multicollinearity happening (which most likely, is the case). I believe that if I. So yeah, I need to do this. And then the discussion would be about whether the final model is "accurate" and "generalizable" (I believe whether it is scalable/tunable - does not play a role here.). Conclusions would be: probably not, as it was unable to handle the spikes correctly.
    Also - I think this temporal prediction will go before training on everything - i think it makes it significantly smoother than introducing it earlier (before the extension). Yeah. So, again, I will use it as a navigator.
### What needs to be done:
* Visualisation of the weekend - to argue for using these features
* well - how could i show that interactions or polyfeatures could be useful? Not sure about that, honestly speaking.
* Remember to add the references at the bottom

## Feedback (from salami)

### Introduction
Does it describe the problem, background, and state a clear goal?
A good introduction should:
(1) motivate WHY the problem matters
(2) describe WHAT will be done, and
(3) state a clear GOAL

I am not sure how to hit that 2nd point (apart from the 3rd).
Just write what will be done? I guess so. 

General remarks:
- Lack of purpose
- Why use both a very simple and a more complex method? <=== exactly. simple should guide us; unless it is clearly a goal in itself to compare those two.
- Several groups copy the exercise text's background verbatim rather
than writing their own

### On methods and model descritipino:
Methods and Model description
- Separates single-band and LDA models
clearly
- States assumptions
- Defines all variables
- Why is it a good assumption?
- Nearly all state the assumptions of
normality
- Very few discuss whether it is met and
what the implications of that is
- If it is not a problem, then argue why
- If it is a problem, then come with
suggestions for what to try instead

Yeah, I need to describe the assumptions.

### On discussions and conclusions
Discussion and Conclusions
Are results interpreted, not just reported?
Good discussions and conclusions (~10 reports):
• Explain WHY later training days give lower error (drying changes spectral properties)
• Discuss the two-class limitation (no background class).
• Note edge/casing issues. Suggest future work
• 1 report circles back to the three goals from the intro.
• 1 report discusses calibration and representative training.
• 1 report states a clear hypothesis and evaluates it.
Problems:
- No discussion section at all (at least 5 reports)!
- A couple of reports: Discussion is 3-4 lines.
- Several reports do not explain WHY LDA outperforms threshold.
- Common pattern: restating numbers ('error was 2.1%') without interpretation.
- No conclusion at all

OK, so the main argument would be to avoid the multi-collinearity.
OK, so the problem that I have is: 
- i need to argue for using the 

Could I go like this:
- i want to implement these 2 features 

What is my argument: simply fucking try and see if it works. Use ridge regression because it can handle lots of the features. That would be my argument. This is how I would argue: I do not know which features could be important. Main motivation is that we don't know upfront; we simply try and see what is happening. 
Motivation for the simpler model:
- first-quick check
- motivates use of the more powerful model
My motivation is:
- we have tried a simpler model to quickly see if it works. It did not. So we used the more features. We used ridge because in expectance of the multicollinearity issues arising from many features, we wanted to do this. It also protects from the possible overfitting due to much more features. (though here it is not an issue).

OK, can my argument go like this? I propose yes.
- i motivate use of the cyclic features
- and of the weekend/holiday features

how do you motivate ridge regression from here and so many features? not sure:
- my motivation is: the model is definitely lacking something. I do not perform deeper inspection. We can add more features and have ridge regression protect us from the multicollinearity and possible overfitting.


## Ideas for me to what put in the discussion:
- mention why did the ridge regression outperform the linear regression.
- why training station matter? I believe?

## Lecture 8 (optical flow feedback)
* remember to cite in the text
* create a breadcrumbs/list of contents? I will ask if this can go separately.
So refereces, plus list of contents - pretty important for them it seems.