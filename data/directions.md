Project Objectives

    Document code, process (data acquistion, preparation, exploratory data analysis and statistical testing, modeling, and model evaluation), findings, and key takeaways in a Jupyter Notebook report.

    Create modules (acquire.py, prepare.py) that make your process repeateable.

    Construct a model to predict customer churn using classification techniques.

    Deliver a 5 minute presentation consisting of a high-level notebook walkthrough using your Jupyter Notebook from above; your presentation should be appropriate for your target audience.

    Answer panel questions about your code, process, findings and key takeaways, and model.

Business Goals

    Find drivers for customer churn at Telco. Why are customers churning?

    Construct a ML classification model that accurately predicts customer churn.

    Document your process well enough to be presented or read like a report.

Audience

    Your target audience for your notebook walkthrough is the Codeup Data Science team. This should guide your language and level of explanations in your walkthrough.

Deliverables

You are expected to deliver the following:

    a Jupyter Notebook Report showing process and analysis with the goal of finding drivers for customer churn. This notebook should be commented and documented well enough to be read like a report or walked through as a presentation.

    a README.md file containing the project description with goals, initial hypotheses, a data dictionary, project planning (lay out your process through the data science pipeline), instructions or an explanation of how someone else can recreate your project and findings (What would someone need to be able to recreate your project on their own?), answers to your hypotheses, key findings, recommendations, and takeaways from your project.

    a CSV file with customer_id, probability of churn, and prediction of churn. (1=churn, 0=not_churn). These predictions should be from your best performing model ran on X_test. Note that the order of the y_pred and y_proba are numpy arrays coming from running the model on X_test. The order of those values will match the order of the rows in X_test, so you can obtain the customer_id from X_test and concatenate these values together into a dataframe to write to CSV.

    individual modules, .py files, that hold your functions to acquire and prepare your data.

    a notebook walkthrough presentation with a high-level overview of your project (5 minutes max). You should be prepared to answer follow-up questions about your code, process, tests, model, and findings.

Project Specifications

Why are our customers churning?

To get started, you may want to think about some of the following questions. Think of these as a jumping off point, but do not let them limit you in coming up with your own questions and ideas.

    Are there clear groupings where a customer is more likely to churn?
        What if you consider contract type?
        Is there a tenure value at which month-to-month customers are most likely to churn? 1-year contract customers? 2-year contract customers?
        Do you have any thoughts on what could be going on? (Be sure to state these thoughts not as facts but as untested hypotheses until you test them!). You might want to plot the rate of churn on a line chart where x is the tenure and y is the rate of churn (customers churned/total customers).

    Are there features that indicate a higher likelihood for customer churn?
        How influential are internet service type, phone service type, online security and backup services, senior citizen status, paying more than x% of customers with the same services, etc.?

    Is there a price threshold for specific services beyond which the likelihood of churn increases?
        If so, what is that point and for which service(s)?
        If we looked at churn rate for month-to-month customers after the 12th month and that of 1-year contract customers after the 12th month, are those rates comparable?

Instructions for each stage of the pipeline are broken down below, but in general, make sure you document your work throughout each stage.

You don't need to explain what every line of code is doing, but you should explain what you are doing and why. In addition, you should not present numers in isolation. If your code outputs a number, be sure you give some context to the number.

    For example: If you drop a feature from the dataset, you should explain why you decided to do so, or why that is a reasonable thing to do. If you transform the data in a column, you should explain why you are making that transformation.

Project Planning

Plan -> Acquire -> Prepare -> Explore -> Model & Evaluate -> Deliver

    Describe the project and goals.

    Task out how you will work through the pipeline in as much detail as you need to keep on track.

    Include a data dictionary to provide context for and explain your data.

    Clearly state your starting hypotheses (and add the testing of these to your task list).

Data Acquisition

Plan -> Acquire -> Prepare -> Explore -> Model & Evaluate -> Deliver

In Your acquire.py module:

    Store functions that are needed to acquire data from the customers table from the telco_churn database on the Codeup data science database server; make sure your module contains the necessary imports to run your code. You will want to join some tables as part of your query.

    Your final function should return a pandas DataFrame.

In Your Notebook:

    Import your acquire function from your acquire.py module and use it to acquire your data in your notebook.

    Complete some initial data summarization (.info(), .describe(), .value_counts(), ...).

    Plot distributions of individual variables.

Data Preparation

Plan -> Acquire -> Prepare -> Explore -> Model & Evaluate -> Deliver

In Your prepare.py module

    Store functions that are needed to prepare your data; make sure your module contains the necessary imports to run your code. Your final function should do the following:

        Split your data into train/validate/test.

        Handle Missing Values.

        Handle erroneous data and/or outliers you wish to address.

        Encode variables as needed.

        Create any new features, if you decided to make any for this project.

In Your Notebook

    Explore missing values and document takeaways/action plans for handling them.

        Is 'missing' equivalent to 0 (or some other constant value) in the specific case of this variable?

        Should you replace the missing values with a value it is most likely to represent, like mean/median/mode?

        Should you remove the variable (column) altogether because of the percentage of missing data?

        Should you remove individual observations (rows) with a missing value for that variable?

    Explore data types and adapt types or data values as needed to have numeric represenations of each attribute.

    Create any new features you want to use in your model. Some ideas you might want to explore after securing a MVP:

        Create a new feature that represents tenure in years.

        Create single variables for or find other methods to merge variables representing the information from the following columns:
            phone_service and multiple_lines
            dependents and partner
            streaming_tv & streaming_movies
            online_security & online_backup

    Import your prepare function from your prepare.py module and use it to prepare your data in your notebook.

Data Exploration & Analysis

Plan -> Acquire -> Prepare -> Explore -> Model & Evaluate -> Deliver

In Your Notebook

    Answer key questions, your initial hypotheses from , and figure out the drivers of churn. You are required to run at least 2 statistical tests in your data exploration. Make sure you document your hypotheses, set your alpha before running the tests, and document your findings well.

    Create visualizations and run statistical tests that work toward discovering variable relationships (independent with independent and independent with dependent). The goal is to identify features that are related to churn (your target), identify any data integrity issues, and understand 'how the data works'. If there appears to be some sort of interaction or correlation, assume there is no causal relationship and brainstorm (and document) ideas on reasons there could be correlation.

    For example: We may find that all who have online services also have device protection. In that case, we don't need both of those.

    Summarize your conclusions, provide clear answers to your specific questions, and summarize any takeaways/action plan from the work above.

Below are some questions you might decide to explore in this stage:

    If a group is identified by tenure, is there a cohort or cohorts who have a higher rate of churn than other cohorts?

        For Example: You might plot the rate of churn on a line chart where x is the tenure and y is the rate of churn (customers churned/total customers)

    Are there features that indicate a higher propensity to churn?

        For Example: type of internet service, type of phone service, online security and backup, senior citizens, paying more than x% of customers with the same services, etc.

    Is there a price threshold for specific services where the likelihood of churn increases once price for those services goes past that point? If so, what is that point for what service(s)?

    If we looked at churn rate for month-to-month customers after the 12th month and that of 1-year contract customers after the 12th month, are those rates comparable?

    Controlling for services (phone_id, internet_service_type_id, online_security_backup, device_protection, tech_support, and contract_type_id), is the mean monthly_charges of those who have churned significantly different from that of those who have not churned? (Use a t-test to answer this.)

    How much of monthly_charges can be explained by internet_service_type?

        Hint: Run a correlation test. State your hypotheses and document your findings clearly.

    How much of monthly_charges can be explained by internet_service_type + phone_service_type (0, 1, or multiple lines).

Modeling and Evaluation

Plan -> Acquire -> Prepare -> Explore -> Model & Evaluate -> Deliver

In Your Notebook

    You are required to establish a baseline accuracy to determine if having a model is better than no model and train and compare at least 3 different models. Document these steps well.

    Train (fit, transform, evaluate) multiple models, varying the algorithm and/or hyperparameters you use.

    Compare evaluation metrics across all the models you train and select the ones you want to evaluate using your validate dataframe.

    Feature Selection (optional): Are there any variables that seem to provide limited to no additional information? If so, remove them.

    Based on the evaluation of your models using the train and validate datasets, choose your best model that you will try with your test data, once.

    Test the final model on your out-of-sample data (the testing dataset), summarize the performance, interpret and document your results.

Delivering

Plan -> Acquire -> Prepare -> Explore -> Model & Evaluate -> Deliver

    Introduce yourself and your project goals at the very beginning of your notebook walkthrough.

    Summarize your findings at the beginning like you would for an Executive Summary. Just because you don't have a slide deck for this presentation, doesn't mean you throw out everything you learned from Storytelling.

    Walk us through the analysis you did to answer our questions and that lead to your findings. Relationships should be visualized and takeaways documented. Please clearly call out the questions and answers you are analyzing as well as offer insights and recommendations based on your findings.

        For example: If you find that month-to-month customers churn more, we won't be surprised, but Telco is not getting rid of that plan. The fact that customers churn is not because they can; it's because they can and they are motivated to do so. We want your insights into why they are motivated to do so. We realize you will not be able to do a full causal experiment, but we would like to see some solid evidence of your conclusions.

    Finish with key takeaways, recommendations, and next steps and be prepared to answer questions from the data science team about your project.

    Remember you have a time limit of 5 minutes for your presentation. Make sure you practice your notebook walkthrough keeping this time limit in mind; it will go by very quickly.
